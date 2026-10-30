"""Developer-authored failure reproductions; not model-generated experiment results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from asset_generation.checks import load_model, load_scene_model, measure_model, scale_limits
from asset_prep.errors import ValidationError
from scene_generation.geometry import collision_scan, bounds, owner
from scene_generation.capabilities import repair_mechanism, repair_signature, repeated_repair
from scene_generation.progress import repair_history_context, failed_approaches


def history(action, outcome, *, selection=None, baseline="v", status="evaluated"):
    return {"status":status, "problem_key":"p", "input_key":baseline, "output_key":baseline+"2",
            "action":action, "action_key":repair_signature(action), "outcome":outcome,
            "selection":selection, "comparison":{"outcome":outcome}}


class LoaderTests(unittest.TestCase):
    def test_measurement_keeps_validation(self):
        cases = [
            '<compiler angle="degree"/>',
            '<compiler angle="radian"/><option><flag contact="disable"/></option>',
            '<compiler angle="radian"/><option><flag equality="disable"/></option>',
            '<compiler angle="radian"/><option timestep=".1"/>',
            '<compiler angle="radian" meshdir="../"/>',
            '<compiler angle="radian"/><asset><mesh file="../outside.obj"/></asset>',
            '<compiler angle="radian"/><include file="other.xml"/>',
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for header in cases:
                (root/"model.xml").write_text('<mujoco>'+header+'<worldbody><geom size=".1"/></worldbody></mujoco>')
                for loader in (load_model, load_scene_model, measure_model):
                    with self.subTest(header=header, loader=loader.__name__), self.assertRaises(ValidationError):
                        loader(root)

    def test_body_and_velocity_limits_are_not_bypassed(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"model.xml").write_text('<mujoco><compiler angle="radian"/></mujoco>')
            for bodies, velocities in ((4097,0),(1,2049)):
                fake = SimpleNamespace(ngeom=1,nbody=bodies,nv=velocities)
                with patch('mujoco.MjModel.from_xml_path', return_value=fake):
                    with self.assertRaises(ValidationError):
                        load_scene_model(root, dynamics=True)

    def test_high_dof_scene_can_publish_static_without_enabling_dynamics(self):
        from asset_generation.static_state import geometry_state
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bodies=''.join(f'<body pos="{i*.1} 0 0"><freejoint/><geom size=".01"/></body>' for i in range(342))
            (root/'model.xml').write_text('<mujoco><compiler angle="radian"/><worldbody>'+bodies+'</worldbody></mujoco>')
            model=load_scene_model(root)
            self.assertEqual(model.nv,2052)
            data=mujoco.MjData(model)
            with patch('mujoco.mj_forward',side_effect=AssertionError('must not run dynamics')), \
                 patch('mujoco.mj_step',side_effect=AssertionError('must not advance time')):
                geometry_state(model,data,contacts=True)
            self.assertEqual(data.time,0)
            self.assertTrue(np.isfinite(data.geom_xpos).all())
            measured,scale=measure_model(root,static=True)
            self.assertFalse(scale['dynamics_within_limits'])
            self.assertTrue(scale_limits(measured)['degraded'])
            with self.assertRaises(ValidationError):load_scene_model(root,dynamics=True)
            with self.assertRaises(ValidationError):measure_model(root)
            with self.assertRaises(ValidationError):load_model(root)

    def test_over_50k_publication_recovery_and_preview_share_loader(self):
        from asset_generation.worker import main
        from scene_generation.recovery_geometry import load_scene
        from scene_generation.preview import export_preview
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact=root/"artifact"; artifact.mkdir()
            geoms=''.join(f'<geom name="a__g{i}" size=".001" pos="{i*.01} 0 0"/>' for i in range(50001))
            (artifact/"model.xml").write_text('<mujoco><compiler angle="radian"/><default><geom contype="0" conaffinity="0"/></default><worldbody>'+geoms+'</worldbody></mujoco>')
            (artifact/"scene.json").write_text(json.dumps({"home_qpos":[],"home_ctrl":[]}))
            # Real 50,001-geom compilation through MjSpec avoids quadratic XML name
            # parsing in every unit run. Archived real XML is separately replayed.
            # The precompiled measurement model needs no geom names in this test; naming
            # 50,001 MjSpec nodes introduces unrelated quadratic name registration.
            # Each entry point still parses/validates the full named XML with the shared gate.
            spec=mujoco.MjSpec()
            for i in range(50001):
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[.001,0,0],pos=[i*.01,0,0],contype=0,conaffinity=0)
            model=spec.compile()
            with patch('mujoco.MjModel.from_xml_path', return_value=model):
                self.assertEqual(load_scene_model(artifact).ngeom,50001)
                with self.assertRaises(ValidationError):
                    load_model(artifact, scope="scene")
                measured, scale=measure_model(artifact)
                self.assertEqual(scale["ngeom"],50001)
                self.assertTrue(scale_limits(model)["degraded"])
                source=root/"input";source.mkdir();output=root/"output";output.mkdir()
                identity="a"*64
                (source/"job.json").write_text(json.dumps({"implementation":identity,"admission":True,
                    "plan":{"jobs":[]},"environment":{},"limits":{"validation_mode":"static"},"contract":{}}))
                args=['worker','scene','--input',str(source),'--output',str(output),'--implementation',identity]
                with patch('sys.argv',args), patch('asset_prep.implementation.verify'), \
                     patch('scene_generation.assembly.assemble',return_value=artifact), contextlib.redirect_stdout(io.StringIO()):
                    main()
                self.assertTrue(json.loads((output/'report.json').read_text(encoding='utf-8'))["assembly_readiness"]["loadable"])
                with patch('asset_generation.static_state.geometry_state'):
                    recovered,_,_=load_scene(artifact)
                    self.assertEqual(recovered.ngeom,50001)
                def render(model, poses, dest, **kwargs):
                    self.assertEqual(model.ngeom,50001);Path(dest).mkdir();return []
                with patch('scene_generation.preview.render_views', side_effect=render):
                    self.assertTrue(export_preview(artifact,root/'preview',{}).is_file())


class WorkBudgetTests(unittest.TestCase):
    def test_actual_budget_survives_success_and_timeout_reports(self):
        from asset_generation.sandbox import ExecutionBackend, BackendExecutionError
        from asset_generation.contracts import GenerationConfig
        from scene_generation.contracts import SceneConfig
        class Backend(ExecutionBackend):
            config=GenerationConfig()
            fail=False
            def job_identity(self):return {'implementation':'a'*64}
            def _run(self,mode,mounts,output):
                self.last_work_budget={'seconds':960}
                if self.fail:raise BackendExecutionError('developer timeout fixture')
                (output/'report.json').write_text('{}')
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);backend=Backend();backend.fail=fail
                limits=SceneConfig(build_timeout_seconds=120)
                with patch('asset_generation.sandbox.checked_report',return_value={}):
                    report,_=backend.assemble_and_check({'instances':[]},{},{},root,root,limits,{})
                self.assertEqual(report['execution_timing']['budget_seconds'],960)
                self.assertEqual(report['execution_timing']['base_budget_seconds'],120)
                self.assertEqual(limits.build_timeout_seconds,120)

    def test_asset_work_retains_original_limit_without_reading_job(self):
        from asset_generation.local import work_budget
        self.assertEqual(work_budget('build', Path('missing'), 120)['seconds'],120)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'job.json').write_text('{"scope":"asset"}')
            self.assertEqual(work_budget('validate', root, 120)['seconds'],120)

    def test_repeated_instances_count_and_two_assembly_passes_are_bounded(self):
        from asset_generation.local import work_budget
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'robot').mkdir();(root/'components/j').mkdir(parents=True)
            (root/'robot/model.xml').write_text('<mujoco/>')
            (root/'components/j/model.xml').write_text('<mujoco>'+('<geom/>'*6000)+'</mujoco>')
            (root/'components/j/interfaces.json').write_text('{}')
            job={'plan':{'instances':[{'job':'j'},{'job':'j'}]},'environment':{}}
            (root/'job.json').write_text(json.dumps(job))
            with patch('scene_generation.robot_contract.model_file',return_value='model.xml'):
                value=work_budget('scene',root,120)
                self.assertEqual(value['estimated_geoms'],12000)
                self.assertEqual(value['seconds'],288)
                job['plan']['instances']*=10;(root/'job.json').write_text(json.dumps(job))
                self.assertEqual(work_budget('scene',root,120)['seconds'],960)

    def test_saved_scene_inspection_uses_single_pass_and_cap(self):
        from asset_generation.local import work_budget
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'artifact').mkdir()
            (root/'job.json').write_text('{"scope":"scene"}')
            (root/'artifact/model.xml').write_text('<mujoco>'+('<geom/>'*42000)+'</mujoco>')
            self.assertEqual(work_budget('inspect',root,120)['seconds'],504)


class GeometryTests(unittest.TestCase):
    def test_focused_scan_does_not_repeat_unrelated_robot_self_contacts(self):
        from types import SimpleNamespace
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="robot__a" size=".1"/>'
            '<geom name="robot__b" size=".1" pos=".1 0 0"/></worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        data=SimpleNamespace(geom_xpos=d.geom_xpos,geom_xmat=d.geom_xmat,
            contact=[SimpleNamespace(geom1=0,geom2=1,dist=-.1)])
        self.assertEqual(collision_scan(m,data,.0005,only_owners={'station'})['collisions'],[])
        self.assertEqual(len(collision_scan(m,data,.0005,only_owners={'robot'})['collisions']),1)

    def test_visual_geometry_does_not_shift_report_names_or_allowed_pairs(self):
        xml='<mujoco><worldbody><geom name="visual" size=".01" pos="5 0 0" contype="0" conaffinity="0"/>' \
            '<geom name="a__g" type="box" size=".05 .05 .05"/>' \
            '<geom name="b__g" type="box" size=".05 .05 .05" pos=".09 0 0"/></worldbody></mujoco>'
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        hits=collision_scan(model,data,.0005)["collisions"]
        self.assertEqual([{h['geom1'],h['geom2']} for h in hits],[{'a__g','b__g'}])
        self.assertEqual(collision_scan(model,data,.0005,allowed=[['a__g','b__g']])["collisions"],[])

    def test_sweep_candidates_match_exhaustive_reference(self):
        # Compare measured pairs, not only reported penetrations. Include inactive
        # geometry, reordered positions, a plane and focus on either pair endpoint.
        rng=np.random.default_rng(412)
        geoms=['<geom name="floor__g" type="plane" size="2 2 .1"/>']
        for i,p in enumerate(rng.uniform(-.25,.25,(40,3))):
            off='contype="0" conaffinity="0"' if i%7==0 else ''
            geoms.append(f'<geom name="o{i%4}__g{i}" type="box" size=".06 .05 .04" pos="{" ".join(map(str,p))}" {off}/>')
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+''.join(geoms)+'</worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d);tol=.0005
        ids=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        for focus in (None,{'o1'},{'o3'},set()):
            expected=set()
            for a,g in enumerate(ids):
                for h in ids[a+1:]:
                    if owner(m,g)==owner(m,h) or focus is not None and not {owner(m,g),owner(m,h)}&focus:continue
                    if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE or m.geom_type[h]==mujoco.mjtGeom.mjGEOM_PLANE:
                        expected.add(tuple(sorted((g,h))));continue
                    lo,hi=bounds(m,d,g);lo2,hi2=bounds(m,d,h)
                    if (np.linalg.norm(d.geom_xpos[g]-d.geom_xpos[h])<=m.geom_rbound[g]+m.geom_rbound[h]+tol
                            and np.all(hi+tol>=lo2) and np.all(hi2+tol>=lo)):
                        expected.add(tuple(sorted((g,h))))
            seen=set()
            def observe(query,g,h,*args):
                seen.add(tuple(sorted((g,h))));return {"outcome":"measured","distance_m":1}
            with patch('asset_generation.distance.observe',side_effect=observe):
                result=collision_scan(m,d,tol,query=object(),only_owners=focus)
            self.assertEqual(seen,expected)
            self.assertTrue(result['complete'])

    def test_pair_budget_reports_unknown_instead_of_pass(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+''.join(
            f'<geom name="o{i}__g" size=".1"/>' for i in range(3))+'</worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        result=collision_scan(m,d,.0005,max_pairs=1)
        self.assertFalse(result['complete']);self.assertGreater(result['unexamined_pairs'],0)


class RepairHistoryTests(unittest.TestCase):
    def test_assembly_failure_identity_changes_with_component_content(self):
        from types import SimpleNamespace
        from scene_generation.scene_first import publish
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            state={'plan':{'jobs':[{'id':'j'}], 'instances':[]},
                   'jobs':{'j':{'artifact':'artifact','assembly_readiness':{'digest':'v1'}}}, 'stages':[]}
            failure={'status':'error','checks':[], 'execution_failure':{'kind':'backend','error_kind':'backend_execution'}}
            run=SimpleNamespace(root=root,state=state,env={},robot_root=root,limits=SimpleNamespace(build_timeout_seconds=120),contract={},
                implementation_guard=SimpleNamespace(expected='code'),store=Mock(),
                begin=Mock(return_value=root),commit=Mock(return_value={'directory':'stage'}),select_repair_candidate=Mock(return_value=False),
                sandbox=SimpleNamespace(assemble_candidate=Mock(return_value=(failure,None))))
            publish(run,[]);publish(run,[])
            self.assertEqual(state['assembly_failure_summary']['attempts_at_this_size'],2)
            state['jobs']['j']['assembly_readiness']['digest']='v2'
            publish(run,[])
            self.assertEqual(state['assembly_failure_summary']['attempts_at_this_size'],1)

    def test_mechanism_preserves_operation_target_and_mobile_fields(self):
        changes=[{'geometry_changes':[{'job':'j','operation':op,'parameters':{'part':part}}]}
                 for op,part in [('hinge','lid'),('lock','lid'),('hinge','door')]]
        changes += [{'layout':{'tables':[]}}, {'robot_posture':{'lift':.1}},
                    {'end_effectors':[{'arm':'left','mode':'stock'}]}, {'tool_additions':[{'id':'tool'}]}]
        self.assertEqual(len({repair_mechanism(a) for a in changes}),len(changes))
        self.assertNotIn('no_change',[repair_mechanism(a) for a in changes])

    def test_coverage_failure_does_not_disprove_exact_geometry_action(self):
        action={'jobs':['j']};h=history(action,'coverage_lost',selection='baseline_preserved')
        self.assertIsNone(repeated_repair([h],'v','p',repair_signature(action)))

    def test_new_baseline_is_not_blocked_by_old_failures(self):
        action={'jobs':['j']};h=history(action,'regressed')
        self.assertIsNone(repeated_repair([h,h],'new','p',repair_signature(action),mechanism=repair_mechanism(action)))

    def test_improvement_pending_and_rejection_stay_separate(self):
        action={'jobs':['j'],'reason':'free text'}
        h=[history(action,'improved_partial',selection='adopted'), history(action,'regressed',selection='baseline_preserved'),
           history(action,None,status='applied')]
        view=repair_history_context(h,'p','v')
        for key in ('accepted_approaches','pending_approaches','rejected_approaches'):
            self.assertEqual(len(view[key]),1)
            self.assertNotIn('reason',view[key][0]['action'])
        self.assertFalse(view['baseline_restored']['restored'])
        self.assertEqual(len(failed_approaches(h,'p')),1)

    def test_restored_flag_tracks_latest_matching_baseline_only(self):
        h=[history({'jobs':['j']},'regressed',selection='baseline_preserved')]
        self.assertTrue(repair_history_context(h,'p','v')['baseline_restored']['restored'])
        self.assertFalse(repair_history_context(h,'p','new')['baseline_restored']['restored'])
        h.append(history({'jobs':['j']},'improved_partial',selection='adopted'))
        self.assertFalse(repair_history_context(h,'p','v')['baseline_restored']['restored'])

    def test_history_context_remains_bounded(self):
        rows=[history({'jobs':[str(i)]},'regressed') for i in range(30)]
        self.assertEqual(len(repair_history_context(rows,'p','v')['rejected_approaches']),6)

    def test_stage_log_preserves_failure_and_boundaries(self):
        from asset_generation.timing import stage
        log=io.StringIO()
        with contextlib.redirect_stdout(log), self.assertRaisesRegex(RuntimeError,'probe'):
            with stage('scene_compile'):raise RuntimeError('probe')
        records=[json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(records[0]['event'],'stage_started')
        self.assertEqual(records[1]['outcome'],'failed')
        self.assertGreaterEqual(records[1]['seconds'],0)

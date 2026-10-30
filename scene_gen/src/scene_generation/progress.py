"""按实际覆盖范围比较证据增量，不把报告措辞或执行完成当作进展。"""
from asset_generation.check_report import complete, incomplete


def pairs(row, *fields):
    return {tuple(sorted((v['geom1'], v['geom2']))) for field in fields for v in row.get(field, [])
            if v.get('geom1') and v.get('geom2')}


def evidence_progress(previous, updated, capability, request, *, status, failure=None, result=None):
    before = {r['id']: r for r in previous.get('checks', [])}
    after = {r['id']: r for r in updated.get('checks', [])}
    touched = set(request.get('checks', [])) | (set(after) if request.get('full_scene') else set())
    actual = {r['id'] for r in result['checks']} if result is not None else touched
    successful = status == 'completed' and failure is None
    resolved, remaining, confirmed, located, resolved_pairs = [], [], [], [], []
    for key in sorted(touched & set(after)):
        old, new = before.get(key, {}), after[key]
        if incomplete(new):
            remaining.append(key)
        if not successful or key not in actual:
            continue
        if (not old or incomplete(old)) and complete(new):
            resolved.append(key)
        removed = pairs(old, 'coverage_gaps', 'measurement_errors') - pairs(new, 'coverage_gaps', 'measurement_errors')
        if result and result.get('scope') == 'pairs':
            removed &= {tuple(sorted(p)) for p in result['covered_pairs']}
        if removed:
            resolved_pairs.append({'check': key, 'pairs': [list(p) for p in sorted(removed)]})
        added = pairs(new, 'collisions') - pairs(old, 'collisions')
        added_components = {c['body'] for c in new.get('unsupported_components',[])} - {c['body'] for c in old.get('unsupported_components',[])}
        if added or added_components:
            confirmed.append({'check': key, 'confirmed_defects': len(added) + len(added_components)})
        elif new.get('status') == 'fail' and old.get('status') != 'fail':
            confirmed.append({'check': key, 'confirmed_defects': 1})
        old_matches = old.get('localization', {}).get('matched', {})
        matches = {name: instance for name, instance in new.get('localization', {}).get('matched', {}).items()
                   if old_matches.get(name) != instance}
        if matches:
            located.append({'check': key, 'instances': sorted(set(matches.values())), 'matched': matches})
    old_open = {f['id'] for f in previous.get('recovery_failures', []) if f['status'] == 'open'}
    new_faults = {f['id']: f for f in updated.get('recovery_failures', [])}
    discharged = sorted(key for key in old_open if key in new_faults and new_faults[key]['status'] == 'resolved') if successful else []
    undischarged = [{'fault': key, 'capability': f.get('capability'), 'kind': f.get('kind'),
                    'remaining_checks': list(f.get('remaining_checks', [])),
                    'remaining_pairs': list(f.get('remaining_pairs', []))}
                   for key, f in new_faults.items() if f['status'] == 'open']
    return {'capability': capability.id, 'status': status, 'requested_checks': sorted(touched),
            'covered_checks': sorted(actual) if successful else [],
            'resolved_gaps': resolved, 'resolved_pairs': resolved_pairs, 'remaining_gaps': remaining,
            'confirmed_defects': confirmed, 'localization': located, 'discharged_faults': discharged,
            'undischarged_faults': undischarged, 'failure': failure,
            'evidence_changed': bool(resolved or resolved_pairs or confirmed or located or discharged)}


def repair_history_context(history, problem_key, input_key=None, limit=6, target_checks=()):
    """Bounded facts for the planner, separated by assessment and selection."""
    from .capabilities import semantic, repair_mechanism
    view = {"rejected_approaches":[], "accepted_approaches":[], "pending_approaches":[],
            "baseline_restored":{"restored":False}}
    relevant = [h for h in history if h.get("problem_key") == problem_key
                or set(target_checks).intersection(h.get("target_checks",[]))]
    for entry in relevant:
        comparison = entry.get("comparison") or {}
        outcome = comparison.get("outcome") or entry.get("outcome")
        action = semantic(entry.get("action") or {})
        same = input_key is not None and input_key in {entry.get("input_key"), entry.get("output_key")}
        row = {"outcome":outcome, "selection":entry.get("selection"), "action":action,
               "same_baseline":same, "mechanism":repair_mechanism(action)}
        for key in ("new_defects", "resolved_checks", "remaining_checks", "coverage_lost",
                    "improved_checks", "partially_improved_checks"):
            row[key] = comparison.get(key, [])[:6]
        if entry.get("status") != "evaluated":
            bucket = "pending_approaches"
        elif entry.get("selection") == "baseline_preserved" or outcome in {"regressed", "mixed", "coverage_lost"}:
            bucket = "rejected_approaches"
        elif entry.get("selection") == "adopted" or outcome in {"improved", "improved_partial", "improved_layout"}:
            bucket = "accepted_approaches"
        elif outcome == "ineffective":
            bucket = "rejected_approaches"
        else:
            bucket = "pending_approaches"
        view[bucket].append(row)
    if relevant:
        latest = relevant[-1]
        if (latest.get("status") == "evaluated" and latest.get("selection") == "baseline_preserved"
                and input_key is not None and latest.get("input_key") == input_key):
            view["baseline_restored"] = {"restored":True,
                "reason":"最近候选被回退，当前输入与其原基线一致；按具体失败证据选择下一步"}
    for key in ("rejected_approaches", "accepted_approaches", "pending_approaches"):
        view[key] = view[key][-limit:]
    return view


def failed_approaches(history, problem_key, limit=6):
    """Compatibility projection: never describe an adopted improvement as rejected."""
    return repair_history_context(history, problem_key, limit=limit)["rejected_approaches"]


def repair_tasks(report, review, plan, input_key):
    """A derived view of existing facts, not a second writable issue ledger."""
    from .recovery import repairable_rows
    from .diagnostics import localization
    tasks=[]
    for row in repairable_rows(report):
        check=row["id"]
        level="asset_structure" if any(k in check for k in ("material","integrity","interaction","specification")) else "inspect_then_choose"
        tasks.append({"id":"check:"+check,"input_key":input_key,"check_id":check,
            "operation_blocking":check.startswith('operation:'),
            "instances":row.get("instances",[]),"certainty":"confirmed_defect","evidence_incomplete":incomplete(row),
            "repair_level":level,"localization":localization(row),"evidence_source":"feedback.checks[id="+check+"]",
            "targets":{k:row[k][:32] for k in ("unsupported_components","disconnected_components","surface_mismatches","collisions") if row.get(k)},
            "target_coverage":{k:{"included":min(32,len(row[k])),"recorded":len(row[k]),
                "truncated":len(row[k])>32,"source":"feedback.checks[id="+check+"]."+k}
                for k in ("unsupported_components","disconnected_components","surface_mismatches","collisions") if row.get(k)},
            "verification":{"checks":[check],"required":"重新检查原问题、受影响依赖和完整场景；缺行、跳过或未知不算解决"}})
    if review and review.get("verdict")=="repair":
        tasks.append({"id":"scene_visual","input_key":input_key,"instances":review.get("affected_instances",[]),
            "layout_targets":review.get("affected_layout",[]),
            "certainty":"review_reported_defect","findings":review["findings"],"repair_level":"inspect_then_choose",
            "verification":{"required":"重新渲染并审查；机械检查结果不可被视觉结论覆盖"}})
    items=plan.get("instances",[])
    for task in tasks:
        jobs={i["job"] for i in items if i["id"] in task["instances"]}
        task["shared_asset_instances"]=[i["id"] for i in items if i["job"] in jobs]
    return tasks


def defect_units(row):
    """Comparable named defects; never infer improvement from truncated counts."""
    from asset_prep.storage import digest
    from .capabilities import semantic
    units={"pair:"+"|".join(p) for p in pairs(row,"collisions")}
    for field in ("unsupported_components","disconnected_components","surface_mismatches"):
        for item in row.get(field,[]):
            # Several disconnected islands or protruding geoms can share one body.
            # Body-only identity collapsed them and hid real partial repairs.
            if field == "disconnected_components" and item.get("geoms"):
                identity=digest({"body":item.get("body"),"geoms":sorted(item["geoms"])})
            else:
                identity=item.get("geom") or item.get("geom1") or item.get("body") or digest(semantic(item))
            units.add(field+":"+identity)
    return units


def defect_count(row):
    """How many named defects a row reports, independent of truncation of the lists."""
    return len(defect_units(row))


def severity(row):
    values={}
    for hit in row.get("collisions",[]):
        if hit.get("geom1") and hit.get("geom2") and type(hit.get("distance_m")) in (int,float):
            values["pair:"+"|".join(sorted((hit["geom1"],hit["geom2"])))] = max(0,-hit["distance_m"])
    if all(type(row.get(k)) in (int,float) for k in ("actual_m","expected_m")):
        values["dimension:"+str(row["expected_m"])]=abs(row["actual_m"]-row["expected_m"])
    return values


def connectivity_delta(old, new):
    """Compare partitions of the same named material, not hashes of whole islands."""
    a,b=old.get('connectivity'),new.get('connectivity')
    if a and b and a.get('version')==b.get('version')==2:
        names,current=set(a['geoms']),set(b['geoms'])
        if not names<=current or a['rigid_groups'].keys()!=b['rigid_groups'].keys():
            return {'split':[sorted(names-current) or ['rigid_group_changed']], 'merged':[]}
        split,merged=[],[]
        for key,group in a['rigid_groups'].items():
            after=b['rigid_groups'][key]
            owners={n:i for i,g in enumerate(after['confirmed']) for n in g}
            for island in group['confirmed']:
                if any(n not in owners for n in island) or len({owners[n] for n in island})>1:
                    split.append(island)
            for island in after['possible']:
                if not set(island)&names:split.append(island)  # A new floating connector is a new defect.
            for island in after['confirmed']:
                if sum(set(g)<=set(island) for g in group['possible'])>1:merged.append(island)
        return {'split':split,'merged':[] if new.get('unexamined_pairs',0) else merged}
    fields = ('disconnected_components', 'disconnected_count')
    partitions = []
    for row in (old, new):
        items = row.get(fields[0], [])
        if not items or row.get(fields[1], len(items)) != len(items):
            return None
        groups = [frozenset(x.get('geoms', [])) for x in items]
        names = set().union(*groups)
        if not all(groups) or sum(map(len, groups)) != len(names):
            return None
        partitions.append((groups, names))
    (before, names), (after, current) = partitions
    if names != current:
        return None  # Disappearance alone cannot distinguish repair from deletion.
    owners = {name: i for i, group in enumerate(after) for name in group}
    split = [sorted(group) for group in before if len({owners[n] for n in group}) > 1]
    merged = [sorted(group) for group in after if sum(g <= group for g in before) > 1]
    # Possible-connectivity partitions include unknown edges. Only count a merge
    # when its material has no unresolved measurement or skipped pair.
    previous_owner = {name:i for i, group in enumerate(before) for name in group}
    uncertain_bridges = []
    for gap in new.get('coverage_gaps', []):
        ends = {name for name in [gap.get('geom1'), gap.get('geom2'), *gap.get('geoms', [])] if name}
        if len(ends) < 2:
            return {'split':split, 'merged':[]}  # Unlocalized uncertainty cannot prove a merge.
        if ends <= names and len({previous_owner[n] for n in ends}) > 1:
            uncertain_bridges.append(ends)
    proven = [group for group in merged if not any(edge <= set(group) for edge in uncertain_bridges)]
    if old.get('unexamined_pairs', 0) or new.get('unexamined_pairs', 0):
        proven = []
    return {'split': split, 'merged': proven}


def repair_comparison(previous, updated, old_review=None, new_review=None, *, old_plan=None, new_plan=None):
    before={r["id"]:r for r in previous.get("checks",[])}
    after={r["id"]:r for r in updated.get("checks",[])}
    resolved, introduced, lost, improved, persistent, superseded = [], [], [], [], [], []
    improved_partial = []
    old_items = {i["id"]:i for i in (old_plan or {}).get("instances", [])}
    new_items = {i["id"]:i for i in (new_plan or {}).get("instances", [])}
    for key,old in before.items():
        new=after.get(key)
        if new is None and key.startswith("table_boundary:"):
            instance = key.removeprefix("table_boundary:")
            support = after.get("support:" + instance, {})
            # The obligation changes only with an actual table-to-floor placement.
            # A complete floor support measurement must replace the table-specific row.
            if (old_items.get(instance, {}).get("placement") == "table"
                    and new_items.get(instance, {}).get("placement") == "floor"
                    and old_items[instance]["job"] == new_items[instance]["job"]
                    and complete(support)):
                superseded.append({"check":key,"replacement":"support:"+instance,
                    "reason":"实际摆放从工作台改为落地；旧桌边约束不适用，地面承托仍按原公差检查"})
                continue
        if new is None or old.get("status")!="skipped" and new.get("status")=="skipped":
            lost.append(key);continue
        if (old.get('connectivity') or {}).get('version')!=(new.get('connectivity') or {}).get('version'):
            lost.append(key)
        old_obligations={g['obligation'] for g in old.get('coverage_gaps',[]) if g.get('obligation')}
        new_obligations={g['obligation'] for g in new.get('coverage_gaps',[]) if g.get('obligation')}
        if new_obligations-old_obligations:lost.append(key)
        connection=connectivity_delta(old,new) if old.get('connectivity') and new.get('connectivity') else None
        if connection and connection['split']:
            introduced.append({'check':key,'split_material':connection['split']})
        if complete(old) and not complete(new):lost.append(key)
        elif any(new.get(k,0)>old.get(k,0) for k in ("measurement_error_count","unexamined_pairs")) or len(new.get("coverage_gaps",[]))>len(old.get("coverage_gaps",[])):
            lost.append(key)
        elif pairs(new, 'coverage_gaps', 'measurement_errors') - pairs(old, 'coverage_gaps', 'measurement_errors'):
            lost.append(key)
        if incomplete(old) and complete(new):
            improved.append({"check":key,"evidence_gap_closed":True})
        old_severity,new_severity=severity(old),severity(new)
        for name in old_severity.keys() & new_severity.keys():
            if new_severity[name]>old_severity[name]:introduced.append({"check":key,"worsened":name,"before":old_severity[name],"after":new_severity[name]})
            elif new_severity[name]<old_severity[name] and complete(new):improved.append({"check":key,"reduced":name,"before":old_severity[name],"after":new_severity[name]})
        if old.get("status")=="fail" and new.get("status")=="fail" and any(new.get(k,0)>old.get(k,0) for k in ("collision_count","disconnected_count","surface_mismatch_count")):
            introduced.append({"check":key,"added":["defect_count_increased"]})
        if old.get("status")=="fail":
            if new.get("status")=="pass" and complete(new):resolved.append(key)
            else:
                persistent.append(key)
                removed=defect_units(old)-defect_units(new)
                added=defect_units(new)-defect_units(old)
                connectivity = connectivity_delta(old, new)
                if connectivity is not None:
                    removed = {x for x in removed if not x.startswith('disconnected_components:')}
                    added = {x for x in added if not x.startswith('disconnected_components:')}
                    if connectivity['split'] and connection is None:
                        introduced.append({'check':key, 'split_material':connectivity['split']})
                    elif connectivity['merged'] and key not in lost:
                        improved_partial.append({'check':key, 'merged_material':connectivity['merged'],
                                                 'remaining':new.get('disconnected_count')})
                # Named defects that disappeared are real evidence even when the check
                # itself still fails; requiring a fully passing row discarded them.
                if removed and not added and (complete(new) or connection is not None and key not in lost) and not any(new.get(count,0)>len(new.get(field,[])) for count,field in
                        (("collision_count","collisions"),("disconnected_count","disconnected_components"),("surface_mismatch_count","surface_mismatches"))):
                    improved_partial.append({"check":key,"removed":sorted(removed),
                        "remaining":len(defect_units(new))})
                if added:introduced.append({"check":key,"added":sorted(added)})
        elif new.get("status")=="fail":introduced.append({"check":key,"added":sorted(defect_units(new)) or ["confirmed_failure"]})
    for key in after.keys()-before.keys():
        if incomplete(after[key]):lost.append(key)
        if after[key].get("status")=="fail":introduced.append({"check":key,"added":["new_check_failure"]})
    if old_review and old_review.get("verdict") in {"accept","repair"}:
        if not new_review or new_review.get("verdict")=="uncertain":lost.append("scene_visual")
        elif old_review["verdict"]=="repair" and new_review["verdict"]=="accept":resolved.append("scene_visual")
        elif old_review["verdict"]=="accept" and new_review["verdict"]=="repair":introduced.append({"check":"scene_visual","added":new_review.get("affected_instances",[])})
    old_advice=previous.get("layout_assessment",{})
    new_advice=updated.get("layout_assessment",{})
    old_metrics=old_advice.get("compactness",{})
    new_metrics=new_advice.get("compactness",{})
    layout_gain=False
    fields=("table_count","table_area_m2","room_area_m2")
    if all(k in old_metrics and k in new_metrics for k in fields):
        old_targets={(r["instance"],t["target"]):t.get("status") for r in old_advice.get("instances",[]) for t in r.get("targets",[])}
        new_targets={(r["instance"],t["target"]):t.get("status") for r in new_advice.get("instances",[]) for t in r.get("targets",[])}
        # Pareto improvement: fewer tables must not hide a larger room/table area.
        extra=("unoccupied_table_lower_bound_m2", "workspace_span_m2")
        compared=fields+tuple(k for k in extra if k in old_metrics and k in new_metrics)
        coverage_ok=all(k not in old_metrics or k in new_metrics for k in extra)
        layout_gain=(coverage_ok and not (old_metrics.get("geometry_complete") and not new_metrics.get("geometry_complete"))
                     and all(new_metrics[k] <= old_metrics[k]+1e-9 for k in compared)
                     and any(new_metrics[k] < old_metrics[k]-1e-9 for k in compared)
                     and bool(old_targets) and old_targets.keys()==new_targets.keys()
                     and all(new_targets[k]=="plausible" for k,v in old_targets.items() if v=="plausible")
                     and not (old_advice.get("home_clear") and not new_advice.get("home_clear")))
    gain=bool(resolved or improved)
    # A repair that removes some named defects while leaving the rest is progress; the
    # old rule only counted a fully cleared check, so partial success looked like a
    # no-op and was discarded, wasting the whole attempt.
    partial_gain=bool(not gain and improved_partial)
    outcome=("coverage_lost" if lost else "mixed" if introduced and (gain or partial_gain)
             else "regressed" if introduced else "improved" if gain
             else "improved_partial" if partial_gain else "improved_layout" if layout_gain else "ineffective")
    return {"outcome":outcome,"layout_compactness_improved":layout_gain,"resolved_checks":sorted(resolved),"improved_checks":improved,
            "partially_improved_checks":improved_partial,
            "new_defects":introduced,"coverage_lost":sorted(lost),"remaining_checks":sorted(persistent),
            "superseded_checks":superseded,
            "adopt":outcome in {"improved","improved_partial","improved_layout"},
            "rule":"同一检查编号、当前版本完整证据才能解决旧问题；能减少已命名缺陷也算进展；取舍候选保留但不自动替换基线"}

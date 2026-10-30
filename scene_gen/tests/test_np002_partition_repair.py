"""Actual paused NP-002 evidence; mutations below test rejection boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from scene_generation.progress import repair_comparison


class PartitionRepairTests(unittest.TestCase):
    def setUp(self):
        self.fixture=json.loads((Path(__file__).parent/'fixtures/np002_sampler_repair.json').read_text('utf-8'))
        self.old,self.new=self.fixture['before'],self.fixture['after']

    def compare(self):
        return repair_comparison({'checks':[self.old]},{'checks':[self.new]})

    def test_real_merge_is_partial_improvement_with_unknowns_retained(self):
        result=self.compare()
        self.assertEqual(result['outcome'],'improved_partial')
        self.assertTrue(result['adopt'])
        self.assertEqual(result['resolved_checks'],[])
        self.assertFalse(self.new['complete'])
        self.assertEqual(len(self.new['coverage_gaps']),6)

    def test_reverse_real_merge_is_a_regression(self):
        self.old,self.new=self.new,self.old
        self.assertFalse(self.compare()['adopt'])

    def test_deleted_material_is_not_a_connectivity_gain(self):
        self.new['disconnected_components'][0]['geoms'].pop()
        self.assertFalse(self.compare()['adopt'])

    def test_new_unknown_pair_is_coverage_loss_even_at_same_count(self):
        self.new['coverage_gaps'][0]['geom2']='new_unmeasured_part'
        self.assertEqual(self.compare()['outcome'],'coverage_lost')

    def test_unknown_bridge_is_not_a_proven_merge(self):
        gap={'geom1':self.old['disconnected_components'][0]['geoms'][0],
             'geom2':self.old['disconnected_components'][1]['geoms'][0], 'error_kind':'measurement_error'}
        self.old['coverage_gaps'].append(deepcopy(gap));self.new['coverage_gaps'].append(gap)
        self.assertFalse(self.compare()['adopt'])


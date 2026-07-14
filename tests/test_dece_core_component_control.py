from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dece_core import compute_component_adaptive_edit_weight_map


class ComponentAdaptiveControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.edit_map = torch.zeros(1, 1, 6, 8)
        self.edit_map[:, :, 1:3, 1:3] = 1.0
        self.edit_map[:, :, 3:5, 5:7] = 1.0
        self.target_delta = self.edit_map.clone()
        self.current_delta = torch.zeros_like(self.edit_map)
        self.current_delta[:, :, 1:3, 1:3] = 0.8
        self.current_delta[:, :, 3:5, 5:7] = 0.2
        self.target_gap = self.target_delta - self.current_delta

    def compute(self, preserve_drift: float = 0.0):
        return compute_component_adaptive_edit_weight_map(
            current_delta=self.current_delta,
            target_delta=self.target_delta,
            target_gap=self.target_gap,
            edit_map=self.edit_map,
            preserve_drift=preserve_drift,
            rmsgap_mode="normgate",
            rmsgap_dead_zone=0.1,
            rmsgap_preserve_gate_budget=0.5,
            edit_target_progress=0.0,
            edit_target_rms=0.0,
            edit_gain=1.0,
            edit_weight_min=1.0,
            edit_weight_max=2.0,
            component_threshold=0.5,
            component_min_pixels=4,
            normalize_channels=True,
        )

    def test_underedited_component_gets_larger_weight(self) -> None:
        weight_map, stats = self.compute()
        self.assertIsNotNone(weight_map)
        assert weight_map is not None
        self.assertAlmostEqual(float(weight_map[0, 0, 1, 1]), 1.1, places=5)
        self.assertAlmostEqual(float(weight_map[0, 0, 3, 5]), 1.7, places=5)
        self.assertEqual(stats["adaptive_component_count"], 2.0)
        self.assertEqual(stats["adaptive_component_active_count"], 2.0)
        self.assertAlmostEqual(stats["adaptive_component_weight_mean"], 1.4, places=5)

    def test_preserve_budget_disables_all_component_boosts(self) -> None:
        weight_map, stats = self.compute(preserve_drift=0.5)
        self.assertIsNotNone(weight_map)
        assert weight_map is not None
        self.assertTrue(torch.equal(weight_map, torch.ones_like(weight_map)))
        self.assertEqual(stats["adaptive_component_active_count"], 0.0)


if __name__ == "__main__":
    unittest.main()

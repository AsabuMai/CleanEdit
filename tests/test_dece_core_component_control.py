from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dece_core import (
    DeceCoreConfig,
    DeceCoreStepInput,
    compute_component_adaptive_edit_weight_map,
    compute_dece_core_step,
)
from flux.dece_flux_adapter import FluxDeceAdapter


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


class SharedBackendComponentParityTest(unittest.TestCase):
    def test_component_flag_is_active_and_backend_neutral(self) -> None:
        edit_map = torch.zeros(1, 1, 6, 8)
        edit_map[:, :, 1:3, 1:3] = 1.0
        edit_map[:, :, 3:5, 5:7] = 1.0
        preserve_map = 1.0 - edit_map
        target = edit_map.expand(1, 2, 6, 8).clone()
        current = torch.zeros_like(target)
        current[:, :, 1:3, 1:3] = 0.8
        current[:, :, 3:5, 5:7] = 0.2
        zeros = torch.zeros_like(target)
        t_scalar = torch.tensor(0.5)
        config = DeceCoreConfig(
            edit_hedit_guidance_scale=1.0,
            adaptive_clean_control=False,
            adaptive_component_control=True,
            adaptive_rmsgap_mode="normgate",
            adaptive_rmsgap_dead_zone=0.1,
            adaptive_edit_gain=1.0,
            adaptive_edit_weight_min=1.0,
            adaptive_edit_weight_max=2.0,
            masked_rms_channel_normalize=True,
        )
        sd3_out = compute_dece_core_step(
            config,
            DeceCoreStepInput(
                z_t=zeros,
                x_src=zeros,
                x0_src=current,
                x0_tar=target,
                v_src=zeros,
                v_tar=target,
                v_src_edit=zeros,
                t_scalar=t_scalar,
                alpha_t=0.0,
                beta_t=1.0,
                x_src_map=zeros,
                x0_src_map=current,
                x0_tar_map=target,
                base_edit_velocity_map=target,
                edit_map=edit_map,
                preserve_map=preserve_map,
                edit_gate=edit_map,
                preserve_gate=preserve_map,
            ),
        )

        flux = FluxDeceAdapter(packed_h=6, packed_w=8)
        flux_out = flux.compute_step(
            config,
            z_t=flux.map_to_packed(zeros),
            x_src=flux.map_to_packed(zeros),
            x0_src=flux.map_to_packed(current),
            x0_tar=flux.map_to_packed(target),
            v_src=flux.map_to_packed(zeros),
            v_tar=flux.map_to_packed(target),
            v_src_edit=flux.map_to_packed(zeros),
            t_scalar=t_scalar,
            alpha_t=0.0,
            beta_t=1.0,
            edit_gate=flux.map_to_packed(edit_map),
            preserve_gate=flux.map_to_packed(preserve_map),
        )

        self.assertGreater(sd3_out.diagnostics["adaptive_component_active_count"], 0.0)
        self.assertEqual(
            sd3_out.diagnostics["adaptive_component_active_count"],
            flux_out.diagnostics["adaptive_component_active_count"],
        )
        self.assertTrue(
            torch.allclose(
                sd3_out.v_total,
                flux.packed_to_map(flux_out.v_total),
                atol=1e-6,
                rtol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()

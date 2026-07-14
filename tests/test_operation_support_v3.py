from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from operation_support_v3 import (  # noqa: E402
    apply_operation_mask_policy,
    build_above_host_region,
    build_container_interior_region,
    build_core_ring_preserve_masks,
    build_face_accessory_region,
    build_operation_support_v3,
    build_relation_region,
    build_support_candidates,
    build_surface_region,
    compute_clean_disagreement,
    compute_velocity_disagreement,
    default_candidate_for_operation,
    infer_removed_prompt_tokens,
    normalize_within_mask,
    parse_edit_operation,
    postprocess_support,
    resolve_operation_mask_policy,
    save_support_debug,
    support_overlap_metrics,
)
from flux.flux_model_ops import _FluxAttentionCaptureBuffer  # noqa: E402


class OperationSupportV3Test(unittest.TestCase):
    def test_above_host_region_builds_relation_box(self):
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 10:16, 7:13] = 1.0

        relation = build_above_host_region(host, expand_x=0.25, above_height=0.5, overlap_height=0.0)

        self.assertGreater(float(relation[0, 0, 8, 10]), 0.9)
        self.assertEqual(float(relation[0, 0, 13, 10]), 0.0)
        self.assertLess(float(relation.mean()), 0.20)

    def test_relation_above_host_default_is_contact_sized(self):
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 8:16, 6:14] = 1.0

        relation = build_relation_region(
            "above_host",
            host_mask=host,
            grounding_mask=None,
            removed_attention_map=None,
            reference=host,
        )

        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertGreater(float(relation[0, 0, 6, 10]), 0.9)
        self.assertEqual(float(relation[0, 0, 10, 6]), 0.0)
        self.assertLess(float(relation.mean()), float(host.mean()))

    def test_surface_region_keeps_host_center(self):
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 5:17, 4:16] = 1.0

        surface = build_surface_region(host, erode_kernel=1, center_width=0.5, center_height=0.5)

        self.assertGreater(float(surface[0, 0, 11, 10]), 0.9)
        self.assertEqual(float(surface[0, 0, 5, 4]), 0.0)
        self.assertLess(float(surface.mean()), float(host.mean()))

    def test_container_interior_region_is_smaller_than_host(self):
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 4:16, 4:16] = 1.0

        interior = build_container_interior_region(host, center_width=0.5, center_height=0.5)
        relation = build_relation_region(
            "inside_container",
            host_mask=host,
            grounding_mask=None,
            removed_attention_map=None,
            reference=host,
        )

        self.assertGreater(float(interior[0, 0, 10, 10]), 0.9)
        self.assertEqual(float(interior[0, 0, 4, 4]), 0.0)
        self.assertLess(float(interior.mean()), float(host.mean()))
        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertLess(float(relation.mean()), float(host.mean()))

    def test_default_candidate_is_operation_aware(self):
        self.assertEqual(parse_edit_operation("decal"), "add_decal")
        self.assertEqual(parse_edit_operation("recolour_object"), "recolor")
        self.assertEqual(parse_edit_operation("material_only"), "material")
        self.assertEqual(
            default_candidate_for_operation("add_object", relation="above_host", has_relation=True),
            "relation_x_response",
        )
        self.assertEqual(
            default_candidate_for_operation("add_object", relation="on_face", has_relation=True),
            "relation_x_response",
        )
        self.assertEqual(
            default_candidate_for_operation("add_decal", relation="on_surface", has_grounding=True),
            "decal_surface_local_response",
        )
        self.assertEqual(
            default_candidate_for_operation("remove_object", has_grounding=True),
            "seg_only",
        )
        self.assertEqual(
            default_candidate_for_operation("replace", has_grounding=True),
            "seg_only",
        )
        self.assertEqual(
            default_candidate_for_operation("recolor", relation="inside", has_relation=True),
            "relation_only",
        )

    def test_operation_mask_policy_separates_geometry_from_appearance(self):
        add = resolve_operation_mask_policy("add_object")
        material = resolve_operation_mask_policy("material")
        replacement = resolve_operation_mask_policy("replace")

        self.assertEqual(add.expansion_kernel, 5)
        self.assertEqual(material.expansion_kernel, 1)
        self.assertTrue(replacement.include_removed_support)

    def test_removed_token_inference_distinguishes_add_from_replace(self):
        self.assertEqual(
            infer_removed_prompt_tokens(
                "A wine glass filled with beer.",
                "A wine glass filled with a red cocktail.",
                "glass",
            ),
            ["beer"],
        )
        self.assertEqual(
            infer_removed_prompt_tokens(
                "A glass filled with milk.",
                "A glass filled with milk and whipped cream.",
                "glass",
            ),
            [],
        )
        self.assertEqual(
            infer_removed_prompt_tokens(
                "A gray cat wearing a crown.",
                "A gray cat wearing a top hat.",
                "cat",
            ),
            ["crown"],
        )

    def test_flux_capture_reuses_one_buffer_for_host_and_removed_groups(self):
        store = _FluxAttentionCaptureBuffer()
        captured = torch.zeros(1, 4, 3)
        captured[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        captured[0, :, 1] = torch.tensor([3.0, 2.0, 1.0, 0.0])
        store.add_cross(0, captured)

        host = store.aggregate_cross(2, 2, token_indices=[0])
        removed = store.aggregate_cross(2, 2, token_indices=[1])

        self.assertFalse(torch.equal(host, removed))
        self.assertEqual(len(store.cross_maps[0]), 1)

    def test_operation_mask_policy_expands_add_but_not_material(self):
        mask = torch.zeros(1, 1, 12, 12)
        mask[:, :, 5:7, 5:7] = 1.0

        add_edit, _, add_stats = apply_operation_mask_policy(
            mask,
            mask,
            edit_operation="add_object",
        )
        material_edit, _, material_stats = apply_operation_mask_policy(
            mask,
            mask,
            edit_operation="material",
        )

        self.assertGreater(float(add_edit.sum()), float(mask.sum()))
        self.assertTrue(torch.equal(material_edit, mask))
        self.assertEqual(add_stats["support_mask_expansion_kernel"], 5)
        self.assertEqual(material_stats["support_mask_expansion_kernel"], 1)

    def test_replacement_policy_unions_compact_removed_support(self):
        mask = torch.zeros(1, 1, 12, 12)
        mask[:, :, 7:9, 7:9] = 1.0
        removed = torch.zeros_like(mask)
        removed[:, :, 2:4, 2:4] = 1.0

        edit, core, stats = apply_operation_mask_policy(
            mask,
            mask,
            edit_operation="replace",
            removed_attention_map=removed,
        )

        self.assertGreater(float(edit[0, 0, 2, 2]), 0.9)
        self.assertGreater(float(core[0, 0, 2, 2]), 0.9)
        self.assertEqual(stats["support_removed_applied"], 1)

    def test_replacement_policy_rejects_diffuse_removed_support(self):
        mask = torch.zeros(1, 1, 12, 12)
        mask[:, :, 7:9, 7:9] = 1.0
        removed = torch.linspace(0.0, 1.0, 12).view(1, 1, 1, 12).expand_as(mask).clone()

        edit, _, stats = apply_operation_mask_policy(
            mask,
            mask,
            edit_operation="replace",
            removed_attention_map=removed,
        )

        self.assertTrue(torch.equal(edit, mask))
        self.assertEqual(stats["support_removed_applied"], 0)
        self.assertEqual(stats["support_removed_rejected_by_area"], 1)

    def test_face_accessory_region_is_smaller_than_host(self):
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 4:18, 3:17] = 1.0

        face = build_face_accessory_region(host)

        self.assertGreater(float(face[0, 0, 7, 10]), 0.9)
        self.assertEqual(float(face[0, 0, 16, 10]), 0.0)
        self.assertLess(float(face.mean()), float(host.mean()))

    def test_clean_velocity_wrappers_and_core_ring_masks(self):
        x_t = torch.zeros(1, 4, 8, 8)
        source_v = torch.zeros_like(x_t)
        target_v = torch.ones_like(x_t)
        clean = compute_clean_disagreement(x_t, torch.tensor(0.5), source_v, target_v)
        velocity = compute_velocity_disagreement(source_v, target_v)
        self.assertEqual(clean.shape, (1, 1, 8, 8))
        self.assertEqual(velocity.shape, (1, 1, 8, 8))

        core_input = torch.zeros(1, 1, 8, 8)
        core_input[:, :, 3:5, 3:5] = 1.0
        core, ring, preserve = build_core_ring_preserve_masks(core_input, ring_dilate_radius=3)
        self.assertGreater(float(core.sum()), 0.0)
        self.assertGreater(float(ring.sum()), 0.0)
        self.assertLess(float(preserve[0, 0, 4, 4]), 0.1)

    def test_overlap_metrics_reports_iou_coverage_leakage(self):
        pred = torch.zeros(1, 1, 8, 8)
        ref = torch.zeros(1, 1, 8, 8)
        pred[:, :, 2:6, 2:6] = 1.0
        ref[:, :, 3:7, 3:7] = 1.0

        metrics = support_overlap_metrics(pred, ref)

        self.assertGreater(metrics["support_iou"], 0.0)
        self.assertLess(metrics["support_iou"], 1.0)
        self.assertGreater(metrics["support_coverage"], 0.0)
        self.assertGreater(metrics["support_leakage"], 0.0)

    def test_build_operation_support_v3_selects_relation_candidate(self):
        x_t = torch.zeros(1, 4, 20, 20)
        source_v = torch.zeros_like(x_t)
        target_v = torch.zeros_like(x_t)
        target_v[:, :, 7:11, 6:14] = 2.0
        attention = torch.zeros(1, 1, 20, 20)
        attention[:, :, 7:11, 8:12] = 1.0
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 11:17, 7:13] = 1.0

        result = build_operation_support_v3(
            attention_map=attention,
            x_t=x_t,
            t=torch.tensor(0.5),
            source_velocity=source_v,
            target_velocity=target_v,
            host_attention_map=host,
            edit_operation="add_object",
            relation="above_host",
            candidate="operation_default",
            top_percentile=80,
            keep_components=1,
            dilate_radius=1,
            blur_kernel=1,
        )

        self.assertEqual(result.stats["support_mode"], "operation_v3")
        self.assertEqual(result.stats["support_score"], "relation_x_response")
        self.assertIsNotNone(result.relation_map)
        self.assertGreater(float(result.edit_mask[0, 0, 8, 10]), 0.0)

    def test_operation_policy_uses_fixed_expansion_and_all_grounded_instances(self):
        x_t = torch.zeros(1, 4, 20, 20)
        source_v = torch.zeros_like(x_t)
        target_v = torch.ones_like(x_t)
        attention = torch.zeros(1, 1, 20, 20)
        attention[:, :, 3:7, 3:7] = torch.linspace(0.2, 1.0, 16).view(1, 1, 4, 4)
        attention[:, :, 13:17, 13:17] = torch.linspace(0.1, 0.5, 16).view(1, 1, 4, 4)
        grounding = torch.zeros_like(attention)
        grounding[:, :, 3:7, 3:7] = 1.0
        grounding[:, :, 13:17, 13:17] = 1.0

        result = build_operation_support_v3(
            attention_map=attention,
            x_t=x_t,
            t=torch.tensor(0.5),
            source_velocity=source_v,
            target_velocity=target_v,
            grounding_mask=grounding,
            edit_operation="add_object",
            keep_components=1,
            dilate_radius=99,
            blur_kernel=1,
            mask_policy="operation",
        )

        self.assertEqual(result.stats["support_mask_policy_mode"], "operation")
        self.assertEqual(result.stats["support_dilate_radius"], 5)
        self.assertEqual(result.stats["support_per_instance_norm"], 1)
        self.assertEqual(result.stats["support_instance_count"], 2)
        self.assertEqual(result.stats["support_keep_components"], 2)
        self.assertEqual(result.stats["support_instance_min_area_ratio"], 0.005)
        self.assertEqual(result.stats["support_instance_max_erosion"], 4)

    def test_surface_local_normalization_ignores_outside_peaks(self):
        value = torch.zeros(1, 1, 8, 8)
        value[:, :, 1, 1] = 100.0
        value[:, :, 4, 4] = 3.0
        value[:, :, 4, 5] = 5.0
        surface = torch.zeros(1, 1, 8, 8)
        surface[:, :, 4:6, 4:6] = 1.0

        local = normalize_within_mask(value, surface)

        self.assertEqual(float(local[0, 0, 1, 1]), 0.0)
        self.assertGreater(float(local[0, 0, 4, 5]), 0.9)

    def test_decal_candidates_include_surface_local_response(self):
        attention = torch.zeros(1, 1, 8, 8)
        attention[:, :, 4:6, 4:6] = 1.0
        clean = torch.zeros(1, 1, 8, 8)
        clean[:, :, 0, 0] = 10.0
        clean[:, :, 4, 4] = 0.8
        velocity = clean.clone()
        surface = torch.zeros(1, 1, 8, 8)
        surface[:, :, 4:6, 4:6] = 1.0

        candidates = build_support_candidates(attention, clean, velocity, relation_map=surface)

        self.assertIn("relation_only", candidates)
        self.assertIn("decal_surface_local_response", candidates)
        self.assertEqual(float(candidates["decal_surface_local_response"][0, 0, 0, 0]), 0.0)
        self.assertGreater(float(candidates["decal_surface_local_response"][0, 0, 4, 4]), 0.0)

    def test_component_scoring_prefers_relation_overlap(self):
        score = torch.zeros(1, 1, 12, 12)
        score[:, :, 2:4, 2:4] = 0.95
        score[:, :, 8:10, 8:10] = 0.90
        clean = score.clone()
        relation = torch.zeros_like(score)
        relation[:, :, 8:10, 8:10] = 1.0

        _, core, stats = postprocess_support(
            score,
            top_percentile=80,
            min_area_ratio=0.0,
            max_area_ratio=0.2,
            keep_components=1,
            dilate_radius=1,
            blur_kernel=1,
            component_score_map=clean,
            relation_map=relation,
            target_area_ratio=4 / 144,
        )

        self.assertGreater(float(core[0, 0, 8, 8]), 0.0)
        self.assertEqual(float(core[0, 0, 2, 2]), 0.0)
        self.assertEqual(stats["support_component_scoring"], 1)

    def test_binary_candidate_does_not_empty_at_high_percentile(self):
        score = torch.zeros(1, 1, 10, 10)
        score[:, :, 4:7, 4:7] = 1.0

        _, core, _ = postprocess_support(
            score,
            top_percentile=95,
            min_area_ratio=0.0,
            max_area_ratio=0.2,
            keep_components=1,
            dilate_radius=1,
            blur_kernel=1,
        )

        self.assertGreater(float(core.sum()), 0.0)

    def test_save_support_debug_writes_candidate_maps(self):
        with self.subTest("debug output"):
            import tempfile

            x_t = torch.zeros(1, 4, 8, 8)
            source_v = torch.zeros_like(x_t)
            target_v = torch.ones_like(x_t)
            attention = torch.ones(1, 1, 8, 8)
            result = build_operation_support_v3(
                attention_map=attention,
                x_t=x_t,
                t=torch.tensor(0.5),
                source_velocity=source_v,
                target_velocity=target_v,
                candidate="attention_x_clean",
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                save_support_debug(result, tmpdir, max_candidates=2)
                self.assertTrue((Path(tmpdir) / "operation_v3_support_score.png").exists())
                self.assertTrue((Path(tmpdir) / "operation_v3_selected_candidate_attention_x_clean.png").exists())
                self.assertTrue((Path(tmpdir) / "selected_candidate_raw.png").exists())
                self.assertTrue((Path(tmpdir) / "selected_candidate_postprocessed.png").exists())
                self.assertTrue((Path(tmpdir) / "M_core.png").exists())
                self.assertTrue((Path(tmpdir) / "M_ring.png").exists())
                self.assertTrue((Path(tmpdir) / "M_preserve.png").exists())
                metadata_path = Path(tmpdir) / "operation_v3_debug_metadata.json"
                self.assertTrue(metadata_path.exists())
                metadata = json.loads(metadata_path.read_text())
                self.assertEqual(metadata["selected_candidate"], "attention_x_clean")


if __name__ == "__main__":
    unittest.main()

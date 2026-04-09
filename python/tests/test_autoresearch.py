from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from python.runners.mini_moe_autoresearch import (
    _best_result,
    _records_and_table_from_state,
    _resume_is_terminal,
    _total_selective_search_space,
    MiniMoeAutoresearchCandidateResult,
    MiniMoeAutoresearchBitmaskTable,
    bitmask_to_mask,
    key_to_mask,
    load_mini_moe_autoresearch_state,
    mask_to_bitmask,
    mask_to_key,
    neighbor_masks,
    top_selective_mask_ids_from_state,
)


class MiniMoeAutoresearchTests(unittest.TestCase):
    def test_mask_key_round_trip(self) -> None:
        mask = (2, 3, 4, 6, 7)
        self.assertEqual(key_to_mask(mask_to_key(mask)), mask)
        self.assertEqual(key_to_mask("none"), ())

    def test_mask_bitmask_round_trip(self) -> None:
        mask = (1, 3, 6, 12)
        self.assertEqual(bitmask_to_mask(mask_to_bitmask(mask), total_layers=16), mask)

    def test_neighbor_masks_cover_add_and_remove_one_layer(self) -> None:
        neighbors = neighbor_masks((2, 4), total_layers=6)
        self.assertEqual(
            neighbors,
            (
                (0, 2, 4),
                (1, 2, 4),
                (2,),
                (2, 3, 4),
                (2, 4, 5),
                (4,),
            ),
        )

    def test_neighbor_masks_exclude_empty_and_full_masks(self) -> None:
        self.assertEqual(neighbor_masks((0,), total_layers=2), ())
        self.assertEqual(neighbor_masks((0, 1), total_layers=3), ((0,), (1,)))

    def test_total_selective_search_space_excludes_reference_and_all_layer(self) -> None:
        self.assertEqual(_total_selective_search_space(1), 0)
        self.assertEqual(_total_selective_search_space(2), 2)
        self.assertEqual(_total_selective_search_space(4), 14)

    def test_resume_terminal_respects_continue_after_success(self) -> None:
        self.assertTrue(_resume_is_terminal("success", stop_on_first_success=True))
        self.assertFalse(_resume_is_terminal("success", stop_on_first_success=False))
        self.assertTrue(_resume_is_terminal("exhausted", stop_on_first_success=False))
        self.assertFalse(_resume_is_terminal("running", stop_on_first_success=True))

    def test_bitmask_table_tracks_pending_and_evaluated_membership(self) -> None:
        mask_a = mask_to_bitmask((2, 4))
        mask_b = mask_to_bitmask((3, 5, 6))
        table = MiniMoeAutoresearchBitmaskTable.from_records(
            total_layers=16,
            evaluated_mask_ids=(mask_a,),
            pending_mask_ids=(mask_a, mask_b),
        )
        self.assertTrue(table.has_evaluated(mask_a))
        self.assertFalse(table.has_pending(mask_a))
        self.assertTrue(table.has_pending(mask_b))
        self.assertEqual(table.pending_count(), 1)

    def test_old_state_schema_migrates_into_bitmask_records(self) -> None:
        old_payload = {
            "baseline_reference": {
                "candidate_name": "reference",
                "mask": [],
                "report_paths": [],
                "avg_final_loss": 3.4,
                "avg_train_toks_per_s": 100.0,
                "avg_overall_toks_per_s": 120.0,
                "avg_peak_process_memory_delta_mb": 1.0,
                "avg_overall_round2_fraction": 0.0,
                "avg_mean_active_round2_fraction": 0.0,
            },
            "baseline_all_layers": {
                "candidate_name": "all_layers",
                "mask": [0, 1, 2, 3],
                "report_paths": [],
                "avg_final_loss": 3.3,
                "avg_train_toks_per_s": 90.0,
                "avg_overall_toks_per_s": 110.0,
                "avg_peak_process_memory_delta_mb": 2.0,
                "avg_overall_round2_fraction": 0.4,
                "avg_mean_active_round2_fraction": 0.4,
            },
            "evaluated_selective": {
                "l1_3": {
                    "candidate_name": "entropy_l1_3",
                    "mask": [1, 3],
                    "report_paths": [],
                    "avg_final_loss": 3.2,
                    "avg_train_toks_per_s": 105.0,
                    "avg_overall_toks_per_s": 125.0,
                    "avg_peak_process_memory_delta_mb": 1.5,
                    "avg_overall_round2_fraction": 0.1,
                    "avg_mean_active_round2_fraction": 0.2,
                }
            },
            "pending_masks": ["l0_2", "l1_2_3"],
        }
        records, table = _records_and_table_from_state(old_payload, total_layers=4)
        self.assertIn(mask_to_bitmask((1, 3)), records)
        self.assertTrue(table.has_evaluated(mask_to_bitmask((1, 3))))
        self.assertTrue(table.has_pending(mask_to_bitmask((0, 2))))
        self.assertTrue(table.has_pending(mask_to_bitmask((1, 2, 3))))

    def test_top_selective_mask_ids_reads_old_schema_state(self) -> None:
        payload = {
            "baseline_reference": {
                "candidate_name": "reference",
                "mask": [],
                "report_paths": [],
                "avg_final_loss": 3.4,
                "avg_train_toks_per_s": 100.0,
                "avg_overall_toks_per_s": 120.0,
                "avg_peak_process_memory_delta_mb": 1.0,
                "avg_overall_round2_fraction": 0.0,
                "avg_mean_active_round2_fraction": 0.0,
            },
            "baseline_all_layers": {
                "candidate_name": "all_layers",
                "mask": [0, 1, 2, 3],
                "report_paths": [],
                "avg_final_loss": 3.3,
                "avg_train_toks_per_s": 90.0,
                "avg_overall_toks_per_s": 110.0,
                "avg_peak_process_memory_delta_mb": 2.0,
                "avg_overall_round2_fraction": 0.4,
                "avg_mean_active_round2_fraction": 0.4,
            },
            "evaluated_selective": {
                "l1_3": {
                    "candidate_name": "entropy_l1_3",
                    "mask": [1, 3],
                    "report_paths": [],
                    "avg_final_loss": 3.2,
                    "avg_train_toks_per_s": 105.0,
                    "avg_overall_toks_per_s": 125.0,
                    "avg_peak_process_memory_delta_mb": 1.5,
                    "avg_overall_round2_fraction": 0.1,
                    "avg_mean_active_round2_fraction": 0.2,
                },
                "l0_2": {
                    "candidate_name": "entropy_l0_2",
                    "mask": [0, 2],
                    "report_paths": [],
                    "avg_final_loss": 3.25,
                    "avg_train_toks_per_s": 106.0,
                    "avg_overall_toks_per_s": 126.0,
                    "avg_peak_process_memory_delta_mb": 1.6,
                    "avg_overall_round2_fraction": 0.12,
                    "avg_mean_active_round2_fraction": 0.22,
                },
            },
            "pending_masks": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                top_selective_mask_ids_from_state(path, total_layers=4, limit=2),
                (mask_to_bitmask((1, 3)), mask_to_bitmask((0, 2))),
            )
            state = load_mini_moe_autoresearch_state(path, total_layers=4)
            self.assertEqual(state["search_table"].pending_count(), 0)

    def test_best_result_prefers_lower_loss_then_higher_speed(self) -> None:
        a = MiniMoeAutoresearchCandidateResult(
            candidate_name="a",
            mask=(0, 4, 6),
            report_paths=(),
            avg_final_loss=3.30,
            avg_train_toks_per_s=250.0,
            avg_overall_toks_per_s=300.0,
            avg_peak_process_memory_delta_mb=0.0,
            avg_overall_round2_fraction=0.1,
            avg_mean_active_round2_fraction=0.3,
        )
        b = MiniMoeAutoresearchCandidateResult(
            candidate_name="b",
            mask=(0, 4, 6, 8),
            report_paths=(),
            avg_final_loss=3.30,
            avg_train_toks_per_s=255.0,
            avg_overall_toks_per_s=305.0,
            avg_peak_process_memory_delta_mb=0.0,
            avg_overall_round2_fraction=0.12,
            avg_mean_active_round2_fraction=0.32,
        )
        c = MiniMoeAutoresearchCandidateResult(
            candidate_name="c",
            mask=(0, 4),
            report_paths=(),
            avg_final_loss=3.29,
            avg_train_toks_per_s=240.0,
            avg_overall_toks_per_s=290.0,
            avg_peak_process_memory_delta_mb=0.0,
            avg_overall_round2_fraction=0.08,
            avg_mean_active_round2_fraction=0.28,
        )
        self.assertIs(_best_result(a, b), b)
        self.assertIs(_best_result(b, c), c)

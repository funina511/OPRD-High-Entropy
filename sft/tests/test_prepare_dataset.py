from __future__ import annotations

import unittest

from sft.prepare_dataset import choose_records, dataset_info, normalize_prompt_messages, to_sft_row


def _record(global_index: int, rollout_index: int, *, valid: bool, correct: bool) -> dict:
    return {
        "global_index": global_index,
        "rollout_index": rollout_index,
        "raw_prompt": [{"role": "user", "content": f"problem {global_index}"}],
        "response": f"solution {global_index}/{rollout_index} \\boxed{{1}}",
        "is_valid": valid,
        "teacher_correct": correct,
        "output_tokens": 12,
    }


class SelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _record(0, 1, valid=True, correct=True),
            _record(0, 0, valid=True, correct=False),
            _record(1, 0, valid=False, correct=False),
        ]

    def test_all_selects_first_rollout(self) -> None:
        selected = choose_records(self.records, selection="all", max_per_prompt=1, keep_empty=False)
        self.assertEqual([(row["global_index"], row["rollout_index"]) for row in selected], [(0, 0), (1, 0)])

    def test_correct_selects_first_correct_rollout(self) -> None:
        selected = choose_records(self.records, selection="correct", max_per_prompt=1, keep_empty=False)
        self.assertEqual([(row["global_index"], row["rollout_index"]) for row in selected], [(0, 1)])

    def test_zero_limit_keeps_all_matches(self) -> None:
        selected = choose_records(self.records, selection="valid", max_per_prompt=0, keep_empty=False)
        self.assertEqual(len(selected), 2)


class ConversionTest(unittest.TestCase):
    def test_prompt_normalization(self) -> None:
        prompt = normalize_prompt_messages(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        )
        self.assertEqual(prompt, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])

    def test_sft_row_ends_with_assistant(self) -> None:
        row = to_sft_row(_record(3, 0, valid=True, correct=True))
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["messages"][-1]["role"], "assistant")

    def test_dataset_info_avoids_custom_openai_converter(self) -> None:
        info = dataset_info("demo", "data.jsonl")
        self.assertEqual(info["demo"]["formatting"], "sharegpt")
        self.assertEqual(info["demo"]["tags"]["role_tag"], "role")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sft.quality import detect_consecutive_repetition, validate_output
from sft.sample_teacher import load_completed, parse_gpu_ids


class QualityTest(unittest.TestCase):
    def test_empty_output_is_invalid(self) -> None:
        result = validate_output("   ")
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "empty")

    def test_boxed_answer_is_valid(self) -> None:
        result = validate_output("Reasoning. Therefore the answer is \\boxed{42}.")
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "ok")

    def test_missing_boxed_answer_is_invalid(self) -> None:
        result = validate_output("The final answer is 42.")
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "no_boxed")

    def test_repeated_long_line_is_invalid(self) -> None:
        repeated = "This is an excessively repeated reasoning line."
        result = validate_output("\\boxed{1}\n" + "\n".join([repeated] * 5))
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "repeated_lines")

    def test_consecutive_repetition_detector(self) -> None:
        block = "abcdefghij" * 5
        self.assertTrue(detect_consecutive_repetition(block * 3))


class GpuIdTest(unittest.TestCase):
    def test_parse_gpu_ids(self) -> None:
        self.assertEqual(parse_gpu_ids("0, 2,3"), [0, 2, 3])

    def test_duplicate_gpu_ids_are_rejected(self) -> None:
        with self.assertRaises(Exception):
            parse_gpu_ids("0,0")


class ResumeStateTest(unittest.TestCase):
    def test_partial_trailing_record_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "worker_0.jsonl"
            valid = {"global_index": 2, "rollout_index": 1, "response": "ok"}
            path.write_text(json.dumps(valid) + "\n{partial", encoding="utf-8")
            completed = load_completed(Path(temporary_dir))
        self.assertEqual(completed[(2, 1)]["response"], "ok")


if __name__ == "__main__":
    unittest.main()

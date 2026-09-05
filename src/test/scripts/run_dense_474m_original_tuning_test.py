from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/models/run_dense_474m_original_tuning.py"
SPEC = importlib.util.spec_from_file_location("dense_474m_original_tuning", SCRIPT)
assert SPEC and SPEC.loader
tuning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tuning)
SUBMIT_SCRIPT = ROOT / "scripts/models/submit_dense_474m_original_tuning.py"
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "submit_dense_474m_original_tuning", SUBMIT_SCRIPT
)
assert SUBMIT_SPEC and SUBMIT_SPEC.loader
submitter = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submitter)


class OriginalTuningPolicyTest(unittest.TestCase):
    def test_bs32_steps_match_existing_original_convention(self) -> None:
        self.assertEqual(tuning.stable_step(1, 32), 6866)
        self.assertEqual(tuning.stable_step(4, 32), 27465)
        self.assertEqual(tuning.total_step(4, 32), 30518)

    def test_bs128_e52_steps(self) -> None:
        self.assertEqual(tuning.stable_step(48, 128), 82397)
        self.assertEqual(tuning.stable_step(52, 128), 89264)
        self.assertEqual(tuning.total_step(52, 128), 99183)

    def test_bs32_selection_requires_strict_improvement_and_retains_baseline_on_tie(self) -> None:
        baseline = {"learningRate": "1e-3", "validationExact": 3.269}
        winner = tuning.choose_bs32(
            [
                {"lr": "5e-4", "validationExact": 3.268},
                {"lr": "2e-3", "validationExact": 3.27},
            ],
            baseline,
        )
        self.assertEqual(winner["selectedLearningRate"], "5e-4")
        self.assertTrue(winner["strictlyImprovesBaseline"])
        tied = tuning.choose_bs32(
            [
                {"lr": "5e-4", "validationExact": 3.269},
                {"lr": "2e-3", "validationExact": 3.27},
            ],
            baseline,
        )
        self.assertEqual(tied["selectedLearningRate"], "1e-3")
        self.assertFalse(tied["strictlyImprovesBaseline"])

    def test_manifest_and_plans(self) -> None:
        manifest = tuning.load_manifest(
            ROOT / "scripts/models/manifests/dense-474m-original-tuning-v1.json"
        )
        bs32 = tuning.plan(manifest, "bs32-probes")
        bs128 = tuning.plan(manifest, "bs128-e52")
        self.assertEqual(bs32["retainedEpochs"], [2, 3, 4])
        self.assertEqual(bs32["evaluationEpoch"], 4)
        self.assertEqual(bs128["retainedEpochs"], [49, 50, 51, 52])
        self.assertEqual(bs128["evaluationEpoch"], 52)

    def test_submission_payload_omits_min_runtime_and_sets_allocator(self) -> None:
        trusted = {
            "version": "v2",
            "tasks": [
                {
                    "arguments": [],
                    "envVars": [
                        {"name": "GIT_REF", "value": "old"},
                        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "old"},
                    ],
                    "resources": {},
                    "context": {"minRuntime": 0},
                }
            ],
        }
        with patch.object(submitter, "base_spec", return_value=trusted):
            for mode, gpu_count in (("bs32-probes", 2), ("bs128-e52", 8)):
                payload = submitter.build_spec(mode, "a" * 40, "urgent")
                task = payload["tasks"][0]
                self.assertEqual(task["resources"]["gpuCount"], gpu_count)
                self.assertNotIn("minRuntime", task["context"])
                allocator = [
                    variable
                    for variable in task["envVars"]
                    if variable.get("name") == "PYTORCH_CUDA_ALLOC_CONF"
                ]
                self.assertEqual(
                    allocator,
                    [{"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"}],
                )


if __name__ == "__main__":
    unittest.main()

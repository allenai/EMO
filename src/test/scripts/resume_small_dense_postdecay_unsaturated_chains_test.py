from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import resume_small_dense_postdecay_unsaturated_chains as resume


class ResumePostDecayUnsaturatedChainsTest(unittest.TestCase):
    def test_only_strict_latest_improvement_resumes(self) -> None:
        improving = {
            "postDecaySourceEpochs": [64, 80, 96],
            "postDecayValidationExact": {"64": 3.354, "80": 3.346, "96": 3.341},
        }
        saturated = {
            "postDecaySourceEpochs": [40, 48, 56],
            "postDecayValidationExact": {"40": 3.072, "48": 3.067, "56": 3.068},
        }
        self.assertTrue(resume.postdecay_is_improving(improving))
        self.assertFalse(resume.postdecay_is_improving(saturated))

    def test_command_preserves_registered_lineage(self) -> None:
        record = {
            "batchSequences": 512,
            "experiment": "01M0K9VRM713XAG1DAPTKWK9V1",
            "lockedWd": "0.3",
            "historicalPreDecayThroughEpoch": 80,
        }
        command = resume.command_for("153m", record, "a" * 40)
        self.assertIn("--postdecay-policy-resume", command)
        self.assertIn("--stop-existing", command)
        self.assertIn("--register", command)
        self.assertEqual(command[command.index("--locked-wd") + 1], "0.3")
        self.assertEqual(
            command[command.index("--historical-predecay-through-epoch") + 1], "80"
        )


if __name__ == "__main__":
    unittest.main()

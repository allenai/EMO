from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/models/run_dense_153m_bs32_lr_wd_hybrid.py"
SPEC = importlib.util.spec_from_file_location("dense_153m_bs32_hybrid", SCRIPT)
assert SPEC and SPEC.loader
hybrid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid)


class HybridPolicyTest(unittest.TestCase):
    def test_steps_match_existing_bs32_convention(self) -> None:
        self.assertEqual(hybrid.stable_step(32), 219726)
        self.assertEqual(hybrid.total_step(32), 244141)
        self.assertEqual(hybrid.stable_step(40), 274657)
        self.assertEqual(hybrid.total_step(40), 305176)
        self.assertEqual(hybrid.stable_step(80), 549316)
        self.assertEqual(hybrid.total_step(80), 610352)

    def test_e40_better_or_tied_continues(self) -> None:
        self.assertEqual(hybrid.decide_probe_route(3.3739, 3.374), "continue_to_e80")
        self.assertEqual(hybrid.decide_probe_route(3.374, 3.374), "continue_to_e80")

    def test_negative_e40_requires_e32_sanity(self) -> None:
        self.assertEqual(hybrid.decide_probe_route(3.375, 3.374), "evaluate_e32_sanity")
        self.assertEqual(hybrid.decide_probe_route(3.375, 3.374, 3.379, 3.378), "select_1e-3")
        self.assertEqual(hybrid.decide_probe_route(3.375, 3.374, 3.378, 3.378), "continue_to_e80")

    def test_best_matched_post_selects_lr_and_retains_baseline_on_tie(self) -> None:
        epochs = [40, 48, 56, 64, 72, 80]
        baseline = {str(epoch): 3.4 - index * 0.001 for index, epoch in enumerate(epochs)}
        better = {epoch: {"validationExact": value} for epoch, value in zip(epochs, [3.39] * 6)}
        selected, _ = hybrid.select_learning_rate(better, baseline)
        self.assertEqual(selected, "5e-4")
        tied = {epoch: {"validationExact": baseline[str(epoch)]} for epoch in epochs}
        selected, decision = hybrid.select_learning_rate(tied, baseline)
        self.assertEqual(selected, "1e-3")
        self.assertEqual(decision["reason"], "baseline_better_or_tied")

    def test_manifest_and_plan(self) -> None:
        manifest = hybrid.load_manifest(
            ROOT / "scripts/models/manifests/dense-153m-bs32-lr-wd-hybrid-v1.json"
        )
        execution_plan = hybrid.plan(manifest)
        self.assertEqual(execution_plan["probe"]["retainedEpochs"], list(range(4, 81, 4)))
        self.assertEqual(execution_plan["probe"]["lateEvaluationEpochs"], [48, 56, 64, 72, 80])
        self.assertEqual(execution_plan["followup"]["evaluationEpochs"], [40, 48, 56, 64, 72, 80])


if __name__ == "__main__":
    unittest.main()

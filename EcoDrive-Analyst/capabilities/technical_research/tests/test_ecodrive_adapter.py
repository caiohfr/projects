from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from capabilities.technical_research.adapters import load_bmw_benchmark_cases


class EcoDriveAdapterTests(unittest.TestCase):
    def test_bmw_benchmark_sample_is_mixed_and_bounded(self):
        root = Path(__file__).resolve().parents[3]
        cases = load_bmw_benchmark_cases(
            root / "artifacts/components/transmission_experiment/EXPERIMENT_SAMPLE_AUDIT.csv",
            root / "artifacts/components/transmission_experiment/TRANSMISSION_CANDIDATE_GROUPS.csv",
            limit=15,
        )
        self.assertEqual(len(cases), 15)
        statuses = {case.previous_identity_status for case in cases}
        self.assertIn("STRICT_CANDIDATE", statuses)
        self.assertIn("FAMILY_ONLY", statuses)
        counts = Counter(case.candidate_group_id for case in cases)
        self.assertGreaterEqual(max(counts.values()), 2)
        for case in cases:
            self.assertNotIn("vde_id", case.request.known_fields)
            self.assertEqual(case.request.known_fields["make"].upper(), "BMW")


if __name__ == "__main__":
    unittest.main()


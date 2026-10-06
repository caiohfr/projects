from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from etl.scripts.sprint_12_transmission_grouping_signal_experiment import (
    DEFAULT_DB,
    GROUP_TERMS,
    PROHIBITED_EXPLANATORY_FIELDS,
    _design_matrices,
    build_fold_assignment,
    load_canonical_source,
    prepare_population,
    run_experiment,
    sha256_file,
    stratified_label_permutation,
)


class Sprint12TransmissionGroupingExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp.name)
        cls.output = cls.temp_path / "artifacts"
        cls.notebook = cls.temp_path / "experiment.ipynb"
        cls.report = cls.temp_path / "result.md"
        cls.hash_before = sha256_file(DEFAULT_DB)
        cls.result = run_experiment(
            DEFAULT_DB,
            cls.output,
            n_permutations=7,
            notebook_path=cls.notebook,
            report_path=cls.report,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_source_database_is_strictly_read_only(self) -> None:
        self.assertEqual(self.hash_before, sha256_file(DEFAULT_DB))
        self.assertEqual(self.result.db_sha256_before, self.result.db_sha256_after)

    def test_carryover_children_never_enter_independent_sample(self) -> None:
        sample = pd.read_csv(self.output / "EXPERIMENT_SAMPLE_AUDIT.csv")
        children = sample["vde_id"].ne(sample["root_vde_id"])
        self.assertEqual(int(children.sum()), self.result.excluded_carryover_count)
        self.assertTrue(sample.loc[children, "included_excluded"].eq("EXCLUDED").all())
        self.assertTrue(
            sample.loc[children, "exclusion_reason"].eq("EXACT_MODEL_YEAR_CARRYOVER_COLLAPSED_TO_ROOT").all()
        )
        self.assertTrue(sample.loc[sample["included_excluded"].eq("INCLUDED"), "vde_id"].eq(
            sample.loc[sample["included_excluded"].eq("INCLUDED"), "root_vde_id"]
        ).all())

    def test_grouping_is_deterministic_and_sentinel_evidence_is_not_strict(self) -> None:
        vdes, runs, fuelcons, _ = load_canonical_source(DEFAULT_DB)
        roots_a, groups_a, _ = prepare_population(vdes, runs, fuelcons)
        roots_b, groups_b, _ = prepare_population(vdes, runs, fuelcons)
        self.assertEqual(groups_a["candidate_group_id"].tolist(), groups_b["candidate_group_id"].tolist())
        ambiguous = groups_a["ratio_evidence_status"].eq("EXPLICIT_AMBIGUOUS_OR_SENTINEL_REVIEW")
        self.assertGreater(int(ambiguous.sum()), 0)
        self.assertTrue(groups_a.loc[ambiguous, "identity_status"].eq("FAMILY_ONLY").all())
        self.assertTrue(groups_a.loc[ambiguous, "usable_ge3"].eq("NO").all())
        self.assertTrue(roots_a.loc[roots_a["model_eligible"], "identity_status"].isin(
            ["DIRECT_MATCH", "STRICT_CANDIDATE"]
        ).all())

    def test_model_design_has_no_target_leakage_and_no_group_quadratic(self) -> None:
        long = pd.read_csv(self.output / "FORCE_CURVE_EXPERIMENT_LONG.csv")
        _, _, baseline_names, augmented_names = _design_matrices(long)
        joined = "|".join(baseline_names)
        for prohibited in PROHIBITED_EXPLANATORY_FIELDS:
            self.assertNotIn(prohibited, joined)
        added = augmented_names[len(baseline_names):]
        self.assertTrue(any(name.endswith(":intercept") for name in added))
        self.assertTrue(any(name.endswith(":speed") for name in added))
        self.assertFalse(any("speed_sq" in name for name in added))
        self.assertEqual(GROUP_TERMS, ("candidate_group_intercept", "candidate_group_linear_speed"))

    def test_grouped_folds_keep_complete_vde_and_application_lineage_together(self) -> None:
        long = pd.read_csv(self.output / "FORCE_CURVE_EXPERIMENT_LONG.csv")
        units = long[["vde_id", "candidate_group_id", "application_lineage"]].drop_duplicates()
        fold_by_vde = build_fold_assignment(units)
        self.assertEqual(len(fold_by_vde), len(units))
        units = units.assign(fold=units["vde_id"].map(fold_by_vde))
        self.assertTrue(units.groupby("application_lineage")["fold"].nunique().eq(1).all())

    def test_stratified_permutation_preserves_label_multiset_per_stratum(self) -> None:
        labels = np.array(["A", "A", "B", "B", "C", "C", "D"])
        strata = np.array(["S1", "S1", "S1", "S1", "S2", "S2", "S3"])
        shuffled = stratified_label_permutation(labels, strata, np.random.default_rng(12))
        for stratum in set(strata.tolist()):
            mask = strata == stratum
            self.assertEqual(sorted(labels[mask].tolist()), sorted(shuffled[mask].tolist()))

    def test_all_required_artifacts_and_executable_notebook_are_created(self) -> None:
        expected = {
            "TRANSMISSION_FIELD_AUDIT.csv",
            "TRANSMISSION_CANDIDATE_GROUPS.csv",
            "EXPERIMENT_SAMPLE_AUDIT.csv",
            "FORCE_CURVE_EXPERIMENT_LONG.csv",
            "MODEL_COMPARISON.csv",
            "GROUP_VALIDATION.csv",
            "PERMUTATION_RESULTS.csv",
            "SENSITIVITY_RESULTS.csv",
        }
        self.assertEqual(expected, {path.name for path in self.output.glob("*.csv")})
        notebook = json.loads(self.notebook.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 10)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(self.notebook), "exec")
        self.assertTrue(self.report.exists())

    def test_notebook_restart_run_all_smoke(self) -> None:
        notebook = json.loads(self.notebook.read_text(encoding="utf-8"))
        namespace: dict = {"__name__": "__notebook_smoke__"}
        old_value = os.environ.get("TRANSMISSION_EXPERIMENT_PERMUTATIONS")
        old_cwd = Path.cwd()
        os.environ["TRANSMISSION_EXPERIMENT_PERMUTATIONS"] = "2"
        os.chdir(DEFAULT_DB.parents[3])
        try:
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), str(self.notebook), "exec"), namespace, namespace)
        finally:
            os.chdir(old_cwd)
            if old_value is None:
                os.environ.pop("TRANSMISSION_EXPERIMENT_PERMUTATIONS", None)
            else:
                os.environ["TRANSMISSION_EXPERIMENT_PERMUTATIONS"] = old_value


if __name__ == "__main__":
    unittest.main()

"""Sprint 10D: the Quick Tech Delta preset catalog (data/quick_tech_deltas.csv)
maps directly onto the existing canonical TechDeltaAssumption contract --
never a second Technology Delta schema (Sec 16/32)."""

import unittest

from src.vde_core.quick_scenario import (
    DomainReadiness,
    EfficiencyQuickInputs,
    PseProvenance,
    QuickScenario,
    TechDeltaAssumption,
    load_quick_tech_delta_catalog,
)
from src.vde_core.technology_delta import normalize_delta_effect_basis


class LoadQuickTechDeltaCatalogTests(unittest.TestCase):
    def test_default_catalog_loads_synthetic_examples(self):
        catalog = load_quick_tech_delta_catalog()
        self.assertGreaterEqual(len(catalog), 1)
        for assumption in catalog.values():
            self.assertIsInstance(assumption, TechDeltaAssumption)

    def test_every_catalog_row_maps_to_a_recognized_canonical_effect_basis(self):
        recognized = {
            "fuel_percent_delta", "pse_percent_delta", "co2_percent_delta",
            "efficiency_multiplier", "metadata_only", "fuel_delta", "pse_delta",
            "pse_multiplier", "co2_delta", "energy_delta", "map_based_effect",
        }
        catalog = load_quick_tech_delta_catalog()
        for assumption in catalog.values():
            self.assertIn(normalize_delta_effect_basis(assumption.effect_basis), recognized)

    def test_catalog_covers_multiple_subsystems(self):
        catalog = load_quick_tech_delta_catalog()
        subsystems = {assumption.affected_subsystem for assumption in catalog.values()}
        self.assertGreaterEqual(len(subsystems), 3)

    def test_missing_catalog_file_returns_empty_mapping_not_an_error(self):
        self.assertEqual(load_quick_tech_delta_catalog("does/not/exist.csv"), {})

    def test_catalog_preset_usable_directly_as_efficiency_quick_input(self):
        catalog = load_quick_tech_delta_catalog()
        preset = catalog["TECH-QA-001"]
        scenario = QuickScenario(
            source_identity="fc:1",
            slot=1,
            efficiency_inputs=EfficiencyQuickInputs(technology_deltas=(preset,)),
        )
        self.assertEqual(scenario.efficiency_inputs.technology_deltas[0].name, "Improved ESS efficiency")

    def test_metadata_only_preset_carries_no_hidden_magnitude(self):
        catalog = load_quick_tech_delta_catalog()
        metadata_only_preset = catalog["TECH-QA-008"]
        self.assertEqual(normalize_delta_effect_basis(metadata_only_preset.effect_basis), "metadata_only")
        self.assertEqual(metadata_only_preset.effect_value, 0.0)


class CustomTechDeltaMapsToCanonicalContractTests(unittest.TestCase):
    def test_custom_assumption_specifies_all_required_fields_explicitly(self):
        custom = TechDeltaAssumption(
            name="Custom calibration tweak",
            effect_basis="pse_percent_delta",
            effect_value=1.2,
            affected_subsystem="calibration",
            source_type="engineering_assumption",
            maturity_level="engineering_assumption",
            confidence="low",
            notes="One-off custom assumption, not from the preset catalog.",
        )
        self.assertEqual(custom.affected_subsystem, "calibration")
        self.assertEqual(normalize_delta_effect_basis(custom.effect_basis), "pse_percent_delta")

    def test_custom_assumption_requires_effect_value_no_default_magnitude(self):
        with self.assertRaises(TypeError):
            TechDeltaAssumption(name="Custom", effect_basis="pse_percent_delta")


if __name__ == "__main__":
    unittest.main()

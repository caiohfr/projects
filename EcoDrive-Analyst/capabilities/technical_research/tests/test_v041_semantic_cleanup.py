from __future__ import annotations

import csv

import pytest

from capabilities.technical_research.semantic_cleanup_v041 import (
    V041Provenance,
    calculated_cda,
    derive_transmission_architecture,
    driveline_semantics,
    normalize_gear_ratios,
    numeric_engineering_value,
    provenance_from_application_match,
    select_v041_value,
    transmission_match_status,
)


def test_exact_researched_value_maps_to_exact_provenance():
    assert provenance_from_application_match("EXACT") == "RESEARCHED_EXACT"


@pytest.mark.parametrize("match", ["STRONG", "PARTIAL"])
def test_compatible_broader_evidence_maps_to_approx_provenance(match):
    assert provenance_from_application_match(match) == "RESEARCHED_APPROX"


def test_approx_researched_value_outranks_rule_and_preserves_it():
    selection = select_v041_value(researched_approx="8HP50", rule_estimated="8-speed automatic")
    assert selection.selected_value == "8HP50"
    assert selection.provenance == V041Provenance.RESEARCHED_APPROX.value
    assert selection.rule_estimated_value == "8-speed automatic"


def test_calculated_cda_remains_numeric_and_calculated():
    value = calculated_cda("0.34", "2.29 m²")
    selection = select_v041_value(calculated=value)
    assert value == pytest.approx(0.7786)
    assert selection.provenance == V041Provenance.CALCULATED.value


def test_exact_hardware_code_is_found_exact():
    assert transmission_match_status(exact_code="GA8HP50Z") == "FOUND_EXACT"


def test_family_only_is_found_family():
    assert transmission_match_status(family="8HP50") == "FOUND_FAMILY"


@pytest.mark.parametrize(
    "description,expected_architecture",
    [
        ("Automatic transmission, single-speed with fixed ratio", "SINGLE_SPEED_EV"),
        ("Eight-speed M Steptronic transmission with Drivelogic", "M_STEPTRONIC_WITH_DRIVELOGIC"),
    ],
)
def test_supported_architecture_is_found_family(description, expected_architecture):
    assert derive_transmission_architecture(description) == expected_architecture
    assert transmission_match_status(marketing_description=description) == "FOUND_FAMILY"


def test_no_transmission_information_is_not_found():
    assert transmission_match_status() == "NOT_FOUND"


def test_credible_disagreement_is_conflict():
    assert transmission_match_status(family="8HP50", conflict=True) == "CONFLICT"


def test_ev_source_placeholder_is_preserved_but_not_physical_fdr():
    result = driveline_semantics(
        source_final_drive="1.00", gear_ratios="front 8.774:1; rear 9.374:1",
        architecture="SINGLE_SPEED_EV", drive_type="All Wheel Drive",
    )
    assert result.source_final_drive == 1.0
    assert result.final_drive is None
    assert result.final_drive_semantics == "SOURCE_PLACEHOLDER"
    assert result.physical_fdr_found is False
    assert result.reduction_front == 8.774
    assert result.reduction_rear == 9.374
    assert result.drive_reduction_found is True


@pytest.mark.parametrize("raw,expected", [("2.29 m²", 2.29), ("3.154:1", 3.154)])
def test_numeric_unit_normalization(raw, expected):
    assert numeric_engineering_value(raw) == expected


def test_conventional_gear_ratio_normalization_excludes_reverse():
    raw = '{"I":"5.000","II":"3.200","III":"2.143","IV":"1.720","V":"1.313","VI":"1.000","VII":"0.823","VIII":"0.640","R":"3.478"}'
    assert normalize_gear_ratios(raw) == "5;3.2;2.143;1.72;1.313;1;0.823;0.64"


def test_full_v041_export_is_read_only_and_freezes_benchmark(tmp_path):
    import scripts.run_component_research_enrichment_v041 as runner

    before = runner.db_hashes()
    metrics = runner.run(v04_dir=runner.DEFAULT_V04, output=tmp_path)
    assert runner.db_hashes() == before
    assert metrics["vehicles_total"] == 15
    assert metrics["transmission_not_found"] == 0
    assert metrics["drive_reduction_found"] == 3
    assert metrics["fdr_found"] == 12
    assert metrics["db_unchanged"] is True
    with (tmp_path / "BMW_COMPONENT_RESEARCH_BENCHMARK_V041.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert all("source_urls" not in row for row in rows)
    assert all("parasitic" not in column.lower() for column in rows[0])
    i4 = next(row for row in rows if row["vde_id"] == "10106")
    assert i4["source_final_drive"] == "1.0"
    assert i4["final_drive"] == ""
    assert i4["reduction_rear"] == "11.115"
    assert i4["gear_ratios"] == ""
    evidence_text = (tmp_path / "COMPONENT_RESEARCH_EVIDENCE_V041.csv").read_text(encoding="utf-8-sig")
    assert "2.29 m²" in evidence_text
    assert "historical_internal_parasitic_loss" not in evidence_text

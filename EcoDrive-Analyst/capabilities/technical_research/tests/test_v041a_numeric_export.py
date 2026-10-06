from __future__ import annotations

import csv

import pytest


NUMERIC_FIELDS = (
    "cd", "frontal_area_m2", "cda_m2", "final_drive",
    "source_final_drive", "reduction_front", "reduction_rear",
)
UNIT_MARKERS = ("m²", "mÂ²", "m2", ":1")


def _read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    import scripts.run_component_research_enrichment_v041 as runner

    output = tmp_path_factory.mktemp("component-v041a")
    before = runner.db_hashes()
    metrics = runner.run(
        v04_dir=runner.DEFAULT_V04,
        output=output,
        artifact_suffix="V041A",
        component_version="0.4.1a",
    )
    assert runner.db_hashes() == before
    return output, metrics


@pytest.mark.parametrize(
    "filename",
    [
        "COMPONENT_RESEARCH_ENRICHMENT_V041A.csv",
        "BMW_COMPONENT_RESEARCH_BENCHMARK_V041A.csv",
    ],
)
def test_serialized_numeric_columns_are_directly_float_parseable(export, filename):
    output, _ = export
    for row in _read(output / filename):
        for field in NUMERIC_FIELDS:
            value = row[field].strip()
            if value:
                float(value)


@pytest.mark.parametrize(
    "filename",
    [
        "COMPONENT_RESEARCH_ENRICHMENT_V041A.csv",
        "BMW_COMPONENT_RESEARCH_BENCHMARK_V041A.csv",
    ],
)
def test_serialized_numeric_columns_have_no_unit_suffixes(export, filename):
    output, _ = export
    for row in _read(output / filename):
        for field in NUMERIC_FIELDS:
            value = row[field].lower()
            assert not any(marker.lower() in value for marker in UNIT_MARKERS)


def test_cda_is_recomputed_from_serialized_numeric_inputs(export):
    output, metrics = export
    rows = _read(output / "COMPONENT_RESEARCH_ENRICHMENT_V041A.csv")
    checked = 0
    for row in rows:
        if not row["cd"] or not row["frontal_area_m2"]:
            continue
        checked += 1
        assert float(row["cda_m2"]) == pytest.approx(
            float(row["cd"]) * float(row["frontal_area_m2"]), abs=1e-12
        )
        assert row["cda_method"] == "DIRECT_PRODUCT"
        assert row["cda_m2_provenance"] == "CALCULATED"
    assert checked == 5
    assert metrics["primary_numeric_validation"]["cda_inconsistent_rows"] == 0


def test_known_unit_bearing_source_values_are_numeric_in_final_artifacts(export):
    output, _ = export
    rows = _read(output / "COMPONENT_RESEARCH_ENRICHMENT_V041A.csv")
    by_vde = {row["vde_id"]: row for row in rows}
    assert by_vde["8535"]["frontal_area_m2"] == "2.29"
    assert by_vde["6849"]["frontal_area_m2"] == "2.13"


def test_ev_placeholder_and_reduction_semantics_are_unchanged(export):
    output, metrics = export
    rows = _read(output / "COMPONENT_RESEARCH_ENRICHMENT_V041A.csv")
    i4_rows = [row for row in rows if row["transmission_family"] == "SINGLE_SPEED_EV"]
    assert len(i4_rows) == 3
    assert all(row["source_final_drive"] == "1.0" for row in i4_rows)
    assert all(row["final_drive"] == "" for row in i4_rows)
    assert all(row["final_drive_semantics"] == "SOURCE_PLACEHOLDER" for row in i4_rows)
    assert all(row["reduction_front"] or row["reduction_rear"] for row in i4_rows)
    assert metrics["drive_reduction_found"] == 3
    assert metrics["fdr_found"] == 12


def test_semantic_counts_and_safety_remain_frozen(export):
    output, metrics = export
    rows = _read(output / "COMPONENT_RESEARCH_ENRICHMENT_V041A.csv")
    assert metrics["transmission_exact_found"] == 0
    assert metrics["transmission_family_found"] == 15
    assert metrics["transmission_not_found"] == 0
    assert metrics["transmission_conflicts"] == 0
    assert metrics["numeric_validation_ok"] is True
    assert metrics["db_unchanged"] is True
    assert all(row["canonical_write"] == "DISABLED" for row in rows)

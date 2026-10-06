from __future__ import annotations

import csv
from pathlib import Path

import pytest

from capabilities.technical_research.contracts import (
    ApplicationMatch,
    EvidenceClaim,
    ExtractionMethod,
    SourceRecord,
    SourceTier,
)
from capabilities.technical_research.core.source_classifier import classify_source
from capabilities.technical_research.cross_oem_v05 import (
    VALUE_FIELDS,
    file_sha256,
    match_status,
    transmission_architecture,
)
from capabilities.technical_research.cross_oem_v051 import (
    CrossOemPragmaticSourcePolicy,
    claims_to_curated_evidence,
    enrich_application_v051,
    grouping_identity,
    normalize_architecture,
)
from scripts.run_component_research_enrichment_v051 import (
    DEFAULT_CURATED,
    DEFAULT_SAMPLE,
    FROZEN_BMW,
    FROZEN_BMW_SHA256,
    _coverage,
    run,
)


ROOT = Path(__file__).resolve().parents[3]


def _sample() -> dict[str, object]:
    return {
        "sample_id": "V05-001",
        "vehicle_configuration_id": "CFG-1",
        "vde_id": 1,
        "make": "Ford",
        "model": "Example",
        "model_year": 2025,
        "category": "Car",
        "drive_type": "2-Wheel Drive, Front",
        "transmission_type": "Semi-Automatic",
        "gears": 6,
        "source_final_drive": 3.2,
        "nv_ratio": 22.0,
        "electrification": "ICE",
        "carryover_root_id": 1,
        "selection_reason": "test",
        "oem_group": "FORD_LINCOLN",
    }


def _evidence(field: str, value: str) -> dict[str, str]:
    return {
        "sample_id": "V05-001",
        "make": "Ford",
        "model_pattern": "",
        "year_min": "2025",
        "year_max": "2025",
        "field": field,
        "value": value,
        "application_match": "EXACT",
        "source_url": "https://www.ford.com/technical.pdf",
        "evidence_note": "Explicit test evidence.",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Semi-Automatic", "OTHER"),
        ("Automated Manual", "AUTOMATED_MANUAL"),
        ("7-speed dual-clutch DCT", "DCT"),
        ("Continuously Variable CVT", "CVT"),
        ("single-speed EV reduction gear", "SINGLE_SPEED_EV"),
    ],
)
def test_specific_architecture_normalization_precedes_generic_substrings(
    raw: str, expected: str
) -> None:
    assert normalize_architecture(raw) == expected


def test_epa_semiautomatic_and_automated_manual_are_not_invented_as_auto_or_dct() -> None:
    assert transmission_architecture("Semi-Automatic", 6, "ICE") == "OTHER"
    assert transmission_architecture("Automated Manual", 7, "ICE") == "AUTOMATED_MANUAL"


def test_status_semantics_distinguish_code_family_architecture_and_not_found() -> None:
    assert match_status(code="10R80") == "FOUND_EXACT"
    assert match_status(family="10R80 family") == "FOUND_FAMILY"
    assert match_status(architecture="DCT") == "FOUND_ARCHITECTURE"
    assert match_status() == "NOT_FOUND"


def test_architecture_label_in_family_is_reclassified_without_family_inflation() -> None:
    row, _ = enrich_application_v051(_sample(), [_evidence("transmission_family", "DCT")])
    assert row["transmission_family"] == ""
    assert row["transmission_architecture"] == "DCT"
    assert row["match_status"] == "FOUND_ARCHITECTURE"


def test_real_family_and_exact_code_keep_precedence() -> None:
    family, _ = enrich_application_v051(
        _sample(), [_evidence("transmission_family", "10R80 family")]
    )
    exact, _ = enrich_application_v051(
        _sample(),
        [
            _evidence("transmission_family", "10R80 family"),
            _evidence("transmission_code", "10R80"),
        ],
    )
    assert family["match_status"] == "FOUND_FAMILY"
    assert exact["match_status"] == "FOUND_EXACT"
    assert grouping_identity(exact) == ("10R80", "EXACT_CODE")


def test_architecture_rows_do_not_increment_family_metric() -> None:
    rows = [
        {"oem_group": "X", "match_status": "FOUND_ARCHITECTURE", "review_flag": "REVIEW_SPARSE"},
        {"oem_group": "X", "match_status": "FOUND_FAMILY", "review_flag": "REVIEW_OK"},
        {"oem_group": "X", "match_status": "FOUND_EXACT", "review_flag": "REVIEW_OK"},
    ]
    first = _coverage(rows, [])
    second = _coverage(rows, [])
    assert first == second
    assert first["transmission_exact_found"] == 1
    assert first["transmission_family_found"] == 1
    assert first["transmission_architecture_found"] == 1


def test_rule_architecture_fallback_does_not_fabricate_family() -> None:
    row, audit = enrich_application_v051(_sample(), [])
    architecture = next(item for item in audit if item["field"] == "transmission_architecture")
    assert row["transmission_architecture"] == "OTHER"
    assert row["transmission_family"] == ""
    assert architecture["provenance"] == "RULE_ESTIMATED"


def test_cross_oem_official_technical_sources_are_primary() -> None:
    source = SourceRecord(
        source_id="ford-tech",
        url="https://media.ford.com/content/specifications/example.pdf",
        title="Technical specifications",
        publisher="media.ford.com",
        document_type="pdf",
        tier=SourceTier.UNCLASSIFIED,
    )
    classification = classify_source(source)
    assert classification.policy_tier == SourceTier.TIER_1_PRIMARY


def test_v051_pragmatic_policy_accepts_official_oem_html() -> None:
    source = SourceRecord(
        source_id="lexus-official",
        url="https://pressroom.lexus.com/example-model/",
        title="Example model overview",
        publisher="pressroom.lexus.com",
        document_type="web_page",
        tier=SourceTier.UNCLASSIFIED,
    )
    from capabilities.technical_research.core.source_classifier import source_with_classification

    decision = CrossOemPragmaticSourcePolicy().evaluate(source_with_classification(source))
    assert decision.decision.value == "ACCEPT"
    assert decision.source.tier == SourceTier.TIER_1_PRIMARY


def test_compatible_same_model_variant_is_approx_but_driveline_values_stay_blocked() -> None:
    def claim(field: str, value: object) -> EvidenceClaim:
        return EvidenceClaim(
            field=field,
            value=value,
            normalized_value=value,
            source_id="ford-tech",
            source_tier=SourceTier.TIER_1_PRIMARY,
            evidence_location="page 1",
            evidence_text="Explicit value.",
            extraction_method=ExtractionMethod.MODEL_EXTRACTED,
            extraction_confidence=0.9,
            application_match=ApplicationMatch.MISMATCH,
            source_url="https://media.ford.com/technical.pdf",
            application_context={
                "make": "Ford",
                "model": "Example AWD",
                "model_year": 2025,
                "drive_type": "4x4",
                "transmission_type": "Semi-Automatic",
            },
        )

    sample = _sample() | {"model": "Example AWD"}
    rows = claims_to_curated_evidence(
        sample,
        [
            claim("transmission_type_normalized", "Semi-Automatic"),
            claim("final_drive_ratio", "4.70"),
        ],
    )
    assert [(row["field"], row["value"], row["application_match"]) for row in rows] == [
        ("transmission_architecture", "OTHER", "PARTIAL")
    ]


def test_broader_model_name_does_not_cross_into_distinct_submodel() -> None:
    claim = EvidenceClaim(
        field="transmission_type_normalized",
        value="Manual",
        normalized_value="MANUAL",
        source_id="bronco",
        source_tier=SourceTier.TIER_1_PRIMARY,
        evidence_location="page 1",
        evidence_text="7-speed manual",
        extraction_method=ExtractionMethod.MODEL_EXTRACTED,
        extraction_confidence=0.9,
        application_match=ApplicationMatch.MISMATCH,
        source_url="https://media.ford.com/bronco.pdf",
        application_context={
            "make": "Ford",
            "model": "Bronco",
            "model_year": 2026,
            "transmission_type": "Manual",
        },
    )
    sample = _sample() | {
        "model": "Bronco Sport",
        "model_year": 2026,
        "transmission_type": "Automatic",
    }
    assert claims_to_curated_evidence(sample, [claim]) == []


def test_contract_excludes_private_parasitic_loss_fields() -> None:
    assert not any("parasitic" in field.lower() or "residual" in field.lower() for field in VALUE_FIELDS)


@pytest.mark.skipif(not DEFAULT_SAMPLE.exists(), reason="accepted v0.5 sample unavailable")
def test_offline_batches_merge_without_losing_rows_or_provenance(tmp_path: Path) -> None:
    report = run(
        sample_path=DEFAULT_SAMPLE,
        curated_path=DEFAULT_CURATED,
        output=tmp_path,
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        live=False,
        resume=False,
        retry_empty_live=False,
        retry_sample_ids=frozenset(),
    )
    with (tmp_path / "COMPONENT_RESEARCH_ENRICHMENT_V051.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        enrichment = list(csv.DictReader(handle))
    with (tmp_path / "COMPONENT_VALUE_PROVENANCE_V051.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        provenance = list(csv.DictReader(handle))
    assert len(enrichment) == 50
    assert len(provenance) == 50 * len(VALUE_FIELDS)
    assert report["db_unchanged"] is True
    assert report["live_research_actually_executed"] is False


def test_frozen_bmw_benchmark_is_unchanged() -> None:
    assert file_sha256(FROZEN_BMW) == FROZEN_BMW_SHA256

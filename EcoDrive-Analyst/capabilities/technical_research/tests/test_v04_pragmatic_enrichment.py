from __future__ import annotations

import csv

from capabilities.technical_research.contracts import (
    ApplicationMatch,
    EvidenceClaim,
    ExtractionMethod,
    SourceTier,
)
from capabilities.technical_research.core import match_application
from capabilities.technical_research.pragmatic_enrichment import (
    PragmaticStatus,
    ResearchedSelection,
    ValueProvenance,
    calculated_cda,
    pragmatic_transmission_status,
    select_value,
    selections_for_application,
    transmission_match_identity,
)


def claim(
    field: str,
    value: object,
    *,
    source: str = "source-1",
    match: ApplicationMatch = ApplicationMatch.EXACT,
) -> EvidenceClaim:
    return EvidenceClaim(
        field=field,
        value=value,
        normalized_value=value,
        source_id=source,
        source_tier=SourceTier.TIER_1_PRIMARY,
        evidence_location="table 1",
        evidence_text=f"The source states {field}={value}.",
        extraction_method=ExtractionMethod.STRUCTURED,
        extraction_confidence=0.95,
        application_match=match,
        source_url=f"https://example.test/{source}",
        document_title="Technical data",
        source_classification="PRIMARY_TECHNICAL",
    )


def test_exact_technical_code_returns_found_exact():
    status, code, _ = pragmatic_transmission_status(
        [claim("transmission_hardware_designation", "GA8HP50Z")]
    )
    assert status == PragmaticStatus.FOUND_EXACT
    assert code.value == "GA8HP50Z"


def test_family_only_source_returns_found_family():
    status, _, family = pragmatic_transmission_status(
        [claim("transmission_family", "ZF 8HP50")]
    )
    assert status == PragmaticStatus.FOUND_FAMILY
    assert family.value == "ZF 8HP50"


def test_conflicting_credible_codes_return_conflict():
    status, code, _ = pragmatic_transmission_status(
        [
            claim("transmission_hardware_designation", "GA8HP50Z", source="a"),
            claim("transmission_hardware_designation", "GA8HP51Z", source="b"),
        ]
    )
    assert status == PragmaticStatus.CONFLICT
    assert code.conflict is True


def test_no_result_returns_not_found():
    assert pragmatic_transmission_status([])[0] == PragmaticStatus.NOT_FOUND


def test_cd_times_area_is_calculated_correctly():
    assert calculated_cda("0.25", "2.20 m²") == 0.55


def test_researched_value_takes_precedence_and_rule_estimate_is_preserved():
    researched = ResearchedSelection("8HP50", (claim("transmission_family", "8HP50"),))
    selected = select_value(researched=researched, rule_estimated="8-speed automatic")
    assert selected.selected_value == "8HP50"
    assert selected.provenance == ValueProvenance.RESEARCHED
    assert selected.rule_estimated_value == "8-speed automatic"


def test_rule_estimate_is_selected_only_when_observed_and_researched_are_missing():
    selected = select_value(rule_estimated="0.62")
    assert selected.selected_value == "0.62"
    assert selected.provenance == ValueProvenance.RULE_ESTIMATED


def test_marketing_description_is_not_exact_technical_code():
    status, _, _ = pragmatic_transmission_status(
        [claim("transmission_marketing_description", "8-speed Steptronic")]
    )
    identity, level = transmission_match_identity("", "", "8-speed Steptronic")
    assert status == PragmaticStatus.NOT_FOUND
    assert identity == "8 SPEED STEPTRONIC"
    assert level == "DESCRIPTION_MATCH_ONLY"


def test_rwd_source_for_xdrive_request_is_not_an_exact_application_match():
    result = match_application(
        {"make": "BMW", "model": "430i xDrive Coupe", "model_year": 2021, "drive_type": "AWD"},
        {"make": "BMW", "model": "430i Coupe", "model_year": 2021, "drive_type": "RWD"},
    )
    assert result.match == ApplicationMatch.MISMATCH
    values, status = selections_for_application(
        {"make": "BMW", "model": "430i xDrive Coupe", "model_year": 2021},
        [claim("transmission_hardware_designation", "GA8HP50Z", match=result.match)],
    )
    assert status == PragmaticStatus.NOT_FOUND
    assert values["transmission_code"].selected_value is None


def test_direct_cda_source_precedes_calculated_and_rule_values():
    values, _ = selections_for_application(
        {"make": "BMW", "model": "330i", "model_year": 2021},
        [
            claim("drag_coefficient_cd", "0.25"),
            claim("frontal_area_m2", "2.20"),
            claim("drag_area_cda_m2", "0.53"),
        ],
        {"cda_m2": "0.62"},
    )
    assert values["cda_m2"].selected_value == "0.53"
    assert values["cda_m2"].provenance == ValueProvenance.RESEARCHED
    assert values["cda_m2"].calculated_value == 0.55
    assert values["cda_m2"].rule_estimated_value == "0.62"


def test_internal_parasitic_loss_claim_is_not_a_supported_output_field():
    values, _ = selections_for_application(
        {"make": "BMW", "model": "330i", "model_year": 2021},
        [claim("historical_internal_parasitic_loss", "12.3")],
    )
    assert all("parasitic" not in field for field in values)
    assert all(selection.provenance != ValueProvenance.RESEARCHED for selection in values.values())


def test_review_export_does_not_modify_canonical_db_or_import_internal_losses(tmp_path):
    import scripts.run_component_research_enrichment_v04 as runner

    before = runner.sha256_file(runner.DEFAULT_DB)
    result = runner.run(
        sample=runner.DEFAULT_SAMPLE,
        groups_path=runner.DEFAULT_GROUPS,
        output=tmp_path,
        db_path=runner.DEFAULT_DB,
        defaults_path=runner.DEFAULT_DEFAULTS,
        v022_dir=runner.DEFAULT_V022,
        curated_path=runner.DEFAULT_CURATED,
        model="unused",
        reasoning_effort="unused",
        live=False,
    )
    with (tmp_path / "COMPONENT_RESEARCH_ENRICHMENT_V04.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert runner.sha256_file(runner.DEFAULT_DB) == before
    assert result["db_unchanged"] is True
    assert len(rows) == 15
    assert all("parasitic" not in column.lower() for column in rows[0])

from __future__ import annotations

import json
from pathlib import Path

from capabilities.technical_research.contracts import SourceRecord
from capabilities.technical_research.live_phasea1 import (
    PhaseA1Cache,
    _extract_claim,
    _normalized_evidence_hash,
    cache_key,
    normalize_query,
)
from src.vde_core.component_enrichment_pass1 import Route, _apply_research_route_override
from src.vde_core.technical_research_phasea1 import _complete_roadload_claim, _promotable


def _task(**updates: str) -> dict[str, str]:
    row = {
        "child_task_id": "RT-TEST",
        "make": "HYUNDAI",
        "model": "SANTA FE HYBRID",
        "model_year_min": "2024",
        "model_year_max": "2026",
        "transmission_code_raw": "G4FT-AC",
        "transmission_type_raw": "Automatic",
    }
    row.update(updates)
    return row


def test_research_override_only_enables_existing_supported_route() -> None:
    ordinary = Route("UNRESOLVED_HYBRID_TOPOLOGY", "NONE_V1", "NOT_IDENTIFIABLE", "HEV", "test", ())
    supported = _apply_research_route_override(ordinary, "PARALLEL_HYBRID_TMED_PMSM_6AT")
    unsupported = _apply_research_route_override(ordinary, "HYBRID_POWER_SPLIT_ECVT")
    assert supported.architecture == "HYBRID_PARALLEL_CONVENTIONAL_TRANS"
    assert supported.model_type == "MOSKALIK_2020"
    assert unsupported is ordinary


def test_roadload_promotion_requires_exact_vde_and_complete_target_set_test() -> None:
    claim = {
        "claim_type": "REGULATORY_TARGET_SET_ROADLOAD_STATE",
        "normalized_value": {
            "vde_id": 7,
            "test_number": "TEST-1",
            "target_abc_native": {"a_lbf": 1, "b_lbf_per_mph": 2, "c_lbf_per_mph2": 3},
            "set_abc_native": {
                "Set Coef A (lbf)": 1,
                "Set Coef B (lbf/mph)": 2,
                "Set Coef C (lbf/mph**2)": 3,
            },
        },
    }
    assert _complete_roadload_claim(claim, 7)
    assert not _complete_roadload_claim(claim, 8)
    del claim["normalized_value"]["test_number"]
    assert not _complete_roadload_claim(claim, 7)


def test_partial_scope_never_promotes_outside_model_year() -> None:
    task = _task()
    claim = {
        "child_task_id": "RT-TEST",
        "application_make": "HYUNDAI",
        "model_year_min": 2024,
        "model_year_max": 2026,
        "claim_type": "DRIVETRAIN_ARCHITECTURE",
        "confidence": "HIGH",
        "provenance": "RESEARCHED",
        "normalized_value": "PARALLEL_HYBRID_TMED_PMSM_6AT",
    }
    assert _promotable(claim, task, 2025, 7)[0]
    assert not _promotable(claim, task, 2023, 7)[0]


def test_boundary_unknown_and_approximate_claim_stays_review_only() -> None:
    task = _task(make="MAZDA", model="MAZDA3")
    claim = {
        "child_task_id": "RT-TEST",
        "application_make": "MAZDA",
        "model_year_min": 2024,
        "model_year_max": 2026,
        "claim_type": "TRANSMISSION_AXLE_BOUNDARY",
        "confidence": "LOW",
        "provenance": "RESEARCHED_APPROX",
        "normalized_value": "APPLICATION_UNCONFIRMED",
        "physical_boundary": "BOUNDARY_UNKNOWN",
    }
    allowed, reason = _promotable(claim, task, 2025, 7)
    assert not allowed
    assert reason == "CONFIDENCE_BELOW_PROMOTION_THRESHOLD"


def test_live_extraction_is_grounded_in_fetched_text() -> None:
    source = SourceRecord("src-1", "https://www.hyundai.com/spec", "Santa Fe Hybrid Specifications", "hyundai.com")
    text = "Santa Fe Hybrid uses a transmission-mounted electric device (TMED) with Smartstream 6AT."
    claim = _extract_claim(_task(), source, text, "HASH")
    assert claim is not None
    assert claim["normalized_value"] == "PARALLEL_HYBRID_TMED_PMSM_6AT"
    assert "transmission-mounted" in claim["supporting_excerpt_short"].lower()
    assert _extract_claim(_task(), source, "Unrelated passenger vehicle page.", "HASH") is None


def test_mazda_negative_case_requires_exact_hardware_code() -> None:
    task = _task(make="MAZDA", model="MAZDA3", transmission_code_raw="1PYTGAAA")
    source = SourceRecord("src-2", "https://www.mazda.com/manual", "Transaxle manual", "mazda.com")
    family_only = "Mazda transaxle final drive differential service information."
    assert _extract_claim(task, source, family_only, "HASH") is None
    exact = "Mazda 1PYTGAAA transaxle final drive and differential service information."
    claim = _extract_claim(task, source, exact, "HASH")
    assert claim is not None
    assert claim["confidence"] == "HIGH"


def test_cache_and_evidence_hash_are_deterministic(tmp_path: Path) -> None:
    key = cache_key("DDGS", normalize_query("  A   Query "), "search", {"limit": 5})
    cache = PhaseA1Cache(tmp_path)
    assert cache.get("search", key) is None
    cache.put("search", key, [{"url": "https://example.com"}])
    assert cache.get("search", key) == [{"url": "https://example.com"}]
    row = {"child_task_id": "RT-1", "source_id": "S1", "claim_type": "X", "value": 1}
    assert _normalized_evidence_hash([row]) == _normalized_evidence_hash([json.loads(json.dumps(row))])

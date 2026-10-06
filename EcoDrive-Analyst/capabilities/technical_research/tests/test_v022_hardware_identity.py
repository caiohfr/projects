from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from capabilities.technical_research.contracts import (
    ApplicationMatch,
    EvidenceClaim,
    ExtractionMethod,
    HardwareGroupMember,
    HardwareGroupStatus,
    IdentityConfidence,
    RequestConsistencyStatus,
    SourceTier,
    TechnicalResearchRequest,
)
from capabilities.technical_research.core import (
    audit_research_request,
    consolidate_research_attempts,
    evaluate_evidence,
    evaluate_hardware_group,
    match_application,
    normalize_transmission_identity_claim,
)


def _claim(
    field: str,
    value: str,
    *,
    source: str = "oem-primary",
    tier: SourceTier = SourceTier.TIER_1_PRIMARY,
    match: ApplicationMatch = ApplicationMatch.EXACT,
) -> EvidenceClaim:
    return EvidenceClaim(
        field=field,
        value=value,
        normalized_value=value.upper(),
        source_id=source,
        source_tier=tier,
        evidence_location="page 1 table 1",
        evidence_text=f"{field}: {value}",
        extraction_method=ExtractionMethod.STRUCTURED,
        extraction_confidence=0.99,
        application_match=match,
    )


def _member(
    app: str,
    hardware: str | None,
    confidence: IdentityConfidence,
    *,
    independent: str | None = None,
    marketing: str | None = None,
) -> HardwareGroupMember:
    return HardwareGroupMember(
        application_id=app,
        independent_application_id=independent or app,
        hardware_identity=hardware,
        identity_confidence=confidence,
        marketing_description=marketing,
        source_ids=(f"source-{app}",) if hardware else (),
    )


class IdentitySemanticsTests(unittest.TestCase):
    def test_marketing_description_is_not_hardware_and_cannot_be_direct(self):
        normalized = normalize_transmission_identity_claim(
            _claim("transmission_designation", "8-speed Steptronic Sport")
        )
        self.assertEqual(normalized.field, "transmission_marketing_description")
        result = evaluate_evidence(
            [normalized],
            "transmission_hardware_designation",
            ("transmission_hardware_designation", "transmission_marketing_description"),
        )
        self.assertIsNone(result.candidate.identity)
        self.assertEqual(result.candidate.confidence, IdentityConfidence.UNRESOLVED)

    def test_exact_primary_hardware_designation_is_direct(self):
        normalized = normalize_transmission_identity_claim(
            _claim("transmission_designation", "GA8HP50Z")
        )
        result = evaluate_evidence(
            [normalized], "transmission_hardware_designation"
        )
        self.assertEqual(result.candidate.identity, "GA8HP50Z")
        self.assertEqual(result.candidate.confidence, IdentityConfidence.DIRECT)

    def test_vehicle_oem_is_not_implicitly_component_supplier(self):
        result = evaluate_evidence(
            [_claim("transmission_hardware_designation", "8HP50")],
            "transmission_hardware_designation",
            ("transmission_supplier",),
        )
        self.assertNotIn("transmission_supplier", result.candidate.attributes)
        self.assertIsNone(result.candidate.field_support["transmission_supplier"].value)


class GroupRuleTests(unittest.TestCase):
    def test_singleton_cannot_confirm(self):
        result = evaluate_hardware_group(
            [_member("a", "8HP50", IdentityConfidence.DIRECT)]
        )
        self.assertEqual(result.status, HardwareGroupStatus.PARTIALLY_RESOLVED)

    def test_two_independent_strong_same_hardware_confirm(self):
        result = evaluate_hardware_group(
            [
                _member("a", "8HP50", IdentityConfidence.DIRECT),
                _member("b", "8HP50", IdentityConfidence.STRONG),
            ]
        )
        self.assertEqual(result.status, HardwareGroupStatus.HARDWARE_CONFIRMED)

    def test_duplicate_application_does_not_confirm(self):
        result = evaluate_hardware_group(
            [
                _member("row-a", "8HP50", IdentityConfidence.DIRECT, independent="same"),
                _member("row-b", "8HP50", IdentityConfidence.DIRECT, independent="same"),
            ]
        )
        self.assertEqual(result.n_independent_applications, 1)
        self.assertNotEqual(result.status, HardwareGroupStatus.HARDWARE_CONFIRMED)

    def test_different_supported_hardware_splits(self):
        result = evaluate_hardware_group(
            [
                _member("a", "8HP50", IdentityConfidence.DIRECT),
                _member("b", "8HP51", IdentityConfidence.DIRECT),
            ]
        )
        self.assertEqual(result.status, HardwareGroupStatus.HARDWARE_SPLIT)

    def test_marketing_only_is_descriptive(self):
        result = evaluate_hardware_group(
            [
                _member("a", None, IdentityConfidence.UNRESOLVED, marketing="Steptronic"),
                _member("b", None, IdentityConfidence.UNRESOLVED, marketing="Steptronic"),
            ]
        )
        self.assertEqual(result.status, HardwareGroupStatus.DESCRIPTIVE_ONLY)

    def test_one_description_does_not_describe_a_larger_unresolved_group(self):
        result = evaluate_hardware_group(
            [
                _member("a", None, IdentityConfidence.UNRESOLVED, marketing="Steptronic"),
                _member("b", None, IdentityConfidence.UNRESOLVED),
            ]
        )
        self.assertEqual(result.status, HardwareGroupStatus.UNRESOLVED)

    def test_mixed_resolved_and_unresolved_is_partial(self):
        result = evaluate_hardware_group(
            [
                _member("a", "8HP50", IdentityConfidence.DIRECT),
                _member("b", None, IdentityConfidence.UNRESOLVED),
            ]
        )
        self.assertEqual(result.status, HardwareGroupStatus.PARTIALLY_RESOLVED)


class RequestAuditAndMatchingTests(unittest.TestCase):
    def test_xdrive_model_with_rwd_field_is_conflicting(self):
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields={
                "make": "BMW",
                "model": "430i xDrive Coupe",
                "model_year": 2020,
                "drive_type": "RWD",
            },
        )
        audit = audit_research_request(request)
        self.assertEqual(
            audit.status, RequestConsistencyStatus.CONFLICTING_SOURCE_FIELDS
        )
        self.assertIn("drive_type", audit.conflicting_fields)

    def test_consistent_and_incomplete_requests(self):
        consistent = audit_research_request(
            {"make": "BMW", "model": "330i", "model_year": 2023, "drive_type": "RWD"}
        )
        incomplete = audit_research_request({"make": "BMW", "model": "330i"})
        self.assertEqual(consistent.status, RequestConsistencyStatus.CONSISTENT)
        self.assertEqual(incomplete.status, RequestConsistencyStatus.INCOMPLETE)

    def test_unknown_optional_field_does_not_create_false_conflict(self):
        audit = audit_research_request(
            {
                "make": "BMW",
                "model": "i4 xDrive40 Gran Coupe",
                "model_year": 2025,
                "drive_type": "AWD",
                "electrification": "UNKNOWN",
            }
        )
        self.assertEqual(audit.status, RequestConsistencyStatus.CONSISTENT)

    def test_conflicting_request_drive_is_not_a_hard_veto_or_overwritten(self):
        request = {
            "make": "BMW",
            "model": "430i xDrive Coupe",
            "model_year": 2020,
            "drive_type": "RWD",
        }
        original = dict(request)
        result = match_application(
            request,
            {
                "make": "BMW",
                "model": "430i xDrive Coupe",
                "model_year": 2020,
                "drive_type": "AWD",
            },
        )
        self.assertNotEqual(result.match, ApplicationMatch.MISMATCH)
        self.assertEqual(request, original)
        self.assertIn("drive_type", result.conflicting_request_fields)


class VehicleEnrichmentTests(unittest.TestCase):
    def test_explicit_cd_frontal_area_and_staggered_tires_are_preserved(self):
        result = evaluate_evidence(
            [
                _claim("transmission_hardware_designation", "8HP50"),
                _claim("drag_coefficient_cd", "0.29"),
                _claim("frontal_area_m2", "2.31 m2"),
                _claim("tire_size_front", "225/45 R18"),
                _claim("tire_size_rear", "255/40 R18"),
            ],
            "transmission_hardware_designation",
        )
        self.assertEqual(result.candidate.attributes["drag_coefficient_cd"], "0.29")
        self.assertEqual(result.candidate.attributes["frontal_area_m2"], "2.31 m2")
        self.assertNotEqual(
            result.candidate.attributes["tire_size_front"],
            result.candidate.attributes["tire_size_rear"],
        )

    def test_cd_conflict_does_not_invalidate_hardware_identity(self):
        result = evaluate_evidence(
            [
                _claim("transmission_hardware_designation", "8HP50"),
                _claim("drag_coefficient_cd", "0.29", source="oem-a"),
                _claim("drag_coefficient_cd", "0.31", source="oem-b"),
            ],
            "transmission_hardware_designation",
        )
        self.assertEqual(result.candidate.identity, "8HP50")
        self.assertEqual(result.candidate.confidence, IdentityConfidence.DIRECT)
        self.assertEqual(
            result.candidate.field_support["drag_coefficient_cd"].conflict_status,
            "UNRESOLVED",
        )


class ArtifactConsolidationTests(unittest.TestCase):
    def test_successful_retry_replaces_error_but_attempt_audit_is_preserved(self):
        attempts = [
            {
                "request_id": "r1",
                "attempt_number": 1,
                "status": "ERROR",
                "retry_reason": "",
            },
            {
                "request_id": "r1",
                "attempt_number": 2,
                "status": "SUPPORTED",
                "retry_reason": "MODEL_EXTRACTION_ERROR",
            },
        ]
        final = consolidate_research_attempts(attempts)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["first_status"], "ERROR")
        self.assertEqual(final[0]["final_status"], "SUPPORTED")
        self.assertEqual(final[0]["attempt_count"], 2)


class CanonicalSafetyTests(unittest.TestCase):
    def test_accepted_sprint_12_database_hashes_are_unchanged(self):
        root = Path(__file__).resolve().parents[3]
        expected = {
            "data/db/eco_drive.db": "243aa746e456e68f9944ee140d24af2595af4252d3fdcb418f500c31364d52ab",
            "data/db/eco_drive_qa.db": "243aa746e456e68f9944ee140d24af2595af4252d3fdcb418f500c31364d52ab",
            # Accepted after the completed canonical component-population
            # materialization; Phase A itself remains strictly read-only.
            "data/db/staging/eco_drive_canonical_candidate.db": "8afac44888388452e9ebf85f8162b2bfa233dfcc74ac3c924339f235a0ea0330",
        }
        for relative_path, expected_hash in expected.items():
            path = root / relative_path
            with self.subTest(path=relative_path):
                self.assertTrue(path.exists())
                with path.open("rb") as handle:
                    self.assertEqual(
                        hashlib.file_digest(handle, "sha256").hexdigest(),
                        expected_hash,
                    )


if __name__ == "__main__":
    unittest.main()

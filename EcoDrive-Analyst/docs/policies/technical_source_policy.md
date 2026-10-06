# Technical Source Policy

## Evidence tiers

1. `TIER_1_PRIMARY`: OEM, supplier/manufacturer, regulator, or standards-body
   technical material.
2. `TIER_2_STRONG_SECONDARY`: peer-reviewed, SAE/academic, or authoritative
   technical catalogs.
3. `TIER_3_DISCOVERY_ONLY`: specialist sources useful for finding stronger
   evidence.
4. `TIER_4_WEAK`: forums, enthusiast pages, aggregators, and unsourced pages.

Unclassified sources are rejected. Tier 3 and Tier 4 cannot independently
establish a component identity.

## Application matching

Source quality never overrides an explicit model, year, engine, trim, or drive
mismatch. Current application matches are `EXACT`, `STRONG`, `PARTIAL`,
`MISMATCH`, or `UNKNOWN`; the compatibility aliases `DIRECT` and `AMBIGUOUS`
map to `EXACT` and `UNKNOWN`. Mismatched claims are excluded from resolution.

## Conflicts

- equal application match allows Tier 1 to resolve weaker evidence;
- a better explicit application match may resolve a tier difference;
- disagreeing high-quality claims with equal applicability remain unresolved;
- unresolved conflicts produce `CONFLICTING_EVIDENCE` and preserve all claims.

## Claim requirements

Every claim carries field, raw and normalized values, source identity and tier,
exact location, short evidence text, extraction method, extraction confidence,
and application match. Hardware identity is never inferred from brand or gear
count alone.

Commercial descriptions such as `8-speed Steptronic` are stored as
`transmission_marketing_description`, not hardware designation. A hardware
designation must be an explicitly stated technical transmission/gearbox code.
The document publisher or vehicle OEM is not copied into component supplier or
manufacturer fields without explicit component-role evidence.

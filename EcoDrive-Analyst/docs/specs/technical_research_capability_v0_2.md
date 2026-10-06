# Technical Research Capability v0.2

v0.2 keeps the v0.1 architecture and enables real external evidence without
adding any canonical EcoDrive write path.

Material behavior changes:

- `DDGSSearchProvider` performs credential-free live discovery and emits only
  unclassified source records.
- `classify_source` assigns authority deterministically from host, publisher,
  path, title, and document type before source policy is applied.
- application matching now returns `EXACT`, `STRONG`, `PARTIAL`, `MISMATCH`,
  or `UNKNOWN`; extractor-supplied authority and match labels are ignored.
- claim conflicts are resolved per field. An unresolved secondary attribute
  no longer removes an independently supported transmission identity.
- every candidate attribute has `SUPPORTED`, `PARTIALLY_SUPPORTED`,
  `CONFLICTING`, or `UNKNOWN` status.
- live extraction uses the OpenAI Responses API through the existing
  LangChain boundary. The configured default is `gpt-5.6-terra` with medium
  reasoning and a 2,500-token output ceiling.
- the HTTP fetcher retains public-address validation and can use DDGS text
  extraction as an audited fallback after an HTTP failure.

The BMW benchmark runner is
`scripts/run_technical_research_benchmark_v02.py`. It is review-only, hashes
the production, QA, and staging candidate databases before and after the run,
and writes only under `artifacts/technical_research`.

## v0.2.2 hardware identity contract

v0.2.2 separates `transmission_hardware_designation`, `transmission_family`,
and `transmission_marketing_description`. Marketing language never establishes
hardware identity or `DIRECT`/`STRONG` confidence. Component supplier and
manufacturer are accepted only when the source explicitly states that role.

The request is self-audited before matching. Missing and internally conflicting
fields remain visible; a conflicting structured field is not used as a hard
application veto and is never silently overwritten.

Every target field has a `FieldEvidenceSummary` containing value, normalized
value, support status, source IDs, strongest tier, application match, and
conflict status. Optional Cd, frontal-area, tire, gear-ratio, and final-drive
claims remain independent from transmission identity.

Group validation uses `HARDWARE_CONFIRMED`, `HARDWARE_SPLIT`,
`PARTIALLY_RESOLVED`, `DESCRIPTIVE_ONLY`, and `UNRESOLVED`.
`HARDWARE_CONFIRMED` requires the same externally supported hardware across at
least two independent applications with `DIRECT`/`STRONG` confidence.

The v0.2.2 runner is `scripts/run_technical_research_benchmark_v022.py`. It
preserves every attempt separately and emits exactly one consolidated final row
per benchmark request.

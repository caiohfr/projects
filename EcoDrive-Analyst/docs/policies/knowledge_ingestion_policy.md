# Knowledge Ingestion Policy

## Decisions

- `INGEST`: unique Tier 1/2 technical content with stable source identity;
- `METADATA_ONLY`: duplicate content, discovery material, or content not safe or
  useful to retain in full;
- `EPHEMERAL`: weak material used only in the current run;
- `REJECT`: unclassified, unsafe, irrelevant, inaccessible, or invalid content.

## Identity and deduplication

Canonical URLs remove fragments and tracking parameters. Normalized content is
fingerprinted with SHA256. Either matching content hash or canonical URL counts
as a duplicate. Source metadata retains publisher, title, type, date, revision,
retrieval time, tier, tags, and ingestion status.

## Content retention

The default design favors metadata, locators, hashes, and short evidence
excerpts. Full content must be retained only when licensing and source policy
allow it. Rejected and ephemeral sources are not chunked or vectorized.

## Retrieval

Local evidence is queried before external search. v0.1 supplies deterministic
lexical retrieval and a vectorizer hook. No embedding provider or vector backend
is selected, preventing a new infrastructure choice from being hidden inside
the research agent.

Result caching is keyed by a normalized request fingerprint. `force_refresh`
bypasses a cached result explicitly; cache hits remain visible in provenance.

## Canonical boundary

The evidence store is a separate database. It is not `component_db`,
`component_instance`, `component_resolution`, VDE, Run, or FuelCons. Ingestion
into the evidence store does not approve or persist canonical engineering truth.

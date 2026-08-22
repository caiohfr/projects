# Package 7G - VDE TOTAL/NET Data Contract Hardening

## Status

Completed. Follows Sprint 7 (Database Management).

## Historical problem

Every pre-Sprint-7 (`record_origin = 'LEGACY'`) row in `vde_db` had
`vde_net_mj_per_km` populated and `vde_total_mj_per_km` NULL. The only
historical write paths for that field (the EPA ETL notebook and the legacy
`vde_setup_service` rich-save path) computed it from the full/TOTAL
coastdown roadload (`coast_A_N`/`coast_B_N_per_kph`/`coast_C_N_per_kph2`)
with no transmission-loss subtraction anywhere in the call chain. Under the
current contract (`NET = TOTAL - resolved transmission losses`), that stored
value was physically `VDE_TOTAL`, not `VDE_NET`.

## Delivered contract

- `src/vde_core/vde_net_total_contract.py` classifies every `vde_db` row
  deterministically (`CANONICAL_TOTAL_ONLY`, `CANONICAL_TOTAL_AND_NET`,
  `LEGACY_TOTAL_IN_NET_FIELD`, `AMBIGUOUS_REVIEW`, `INVALID`) and exposes
  `canonical_vde_read()` as the single clean TOTAL/NET read contract. A
  net-only row is only ever reclassified as a legacy swap when
  `record_origin == "LEGACY"` - the only origin with proven historical
  evidence. Every other net-only row is `AMBIGUOUS_REVIEW`, never guessed.
- `src/vde_core/vde_net_total_normalization.py` +
  `scripts/normalize_vde_net_total.py` provide an idempotent, auditable
  preview/apply migration: legacy rows move their NET value into TOTAL and
  null NET; ambiguous rows only get `review_status = 'REVIEW_REQUIRED'`
  flagged, never a guessed value. Every applied change is logged in
  `data_change_log`.
- `vde_db.review_status` (reusing the existing `ReviewStatus` contract)
  surfaces ambiguous rows for manual review instead of silently keeping or
  discarding them.
- Database Management's VDE tab now shows `vde_total_mj_per_km`,
  `vde_net_mj_per_km`, and `review_status` read-only, governed by the
  existing Sprint 7 field-policy (`DERIVED`/`ADVANCED_CORRECTION`) rules.

## Operational boundaries

- New code must read VDE TOTAL/NET through `canonical_vde_read()`, never by
  falling back from one field to the other.
- The normalization never derives NET values or invents transmission losses;
  ambiguous rows stay `REVIEW_REQUIRED` rather than being corrected.
- The real `data/db/eco_drive.db` was backed up before normalization was
  applied (5003/5003 legacy rows corrected, 0 ambiguous, idempotency
  confirmed).
- Underlying roadload/VDE/mass/tire/transmission physics were not changed -
  this package is a storage/data-contract correction only.

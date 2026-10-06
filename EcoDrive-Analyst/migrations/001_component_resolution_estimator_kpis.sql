-- EcoDrive Sprint 12 DB Closure
-- SQLite does not support portable ADD COLUMN IF NOT EXISTS syntax.
-- The import utility checks PRAGMA table_info(component_resolution) before
-- executing each statement below.

ALTER TABLE component_resolution ADD COLUMN estimate_status TEXT;
ALTER TABLE component_resolution ADD COLUMN estimator_version TEXT;
ALTER TABLE component_resolution ADD COLUMN fit_nrmse_pct REAL;
ALTER TABLE component_resolution ADD COLUMN condition_number REAL;
ALTER TABLE component_resolution ADD COLUMN sensitivity_rel_pct REAL;

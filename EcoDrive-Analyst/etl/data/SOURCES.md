# EcoDrive Engineering ETL — Data Sources

This directory contains local public datasets used by the EcoDrive
Engineering ETL.

Large/raw source files are intentionally excluded from Git.

## EPA

- EPA Test Car List — MY2026
- EPA Certified Vehicle Test Results — 2014–Present
- EPA Certified Vehicle Models — 2014–Present
- FuelEconomy.gov — MY2026
- EPA EV-CIS Light-Duty Data Requirements
- EPA EV-CIS Light-Duty Business Rules
- EPA certification application sample documents

## WLTP / Europe

- EEA CO2 Monitoring — 2025 Provisional
- JRC passenger-car technical dataset
- UNECE Regulation No. 154 — WLTP
- EPREL tyre database — online enrichment source

## Data policy

Raw files are immutable source material.

Pipeline transformations go to:

- `processed/`
- `staging/`

The EcoDrive runtime/database is not modified during the initial
Sprint 12 ETL research phase.
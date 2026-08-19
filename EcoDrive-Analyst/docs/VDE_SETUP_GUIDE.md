# VDE Setup Guide

## Purpose

VDE Setup creates traceable vehicle-demand scenarios from a canonical baseline.
It is the current stable engineering product in EcoDrive. It resolves the
scenario before presentation, comparison, and persistence; those later views do
not maintain separate physics calculations.

## Workflow

1. **Baseline & Corrections**: select a canonical baseline and apply optional
   corrections to establish the effective baseline.
2. **Proposal Matrix**: define requested scenarios, their domain changes, and
   Walk From source.
3. **Request Inputs**: select data sources and apply Mass, Tire, Aero,
   Transmission, Brake, Axle & Hubs, and Parasitics inputs.
4. **Preview & Save**: validate the resolved scenarios, compare roadload and
   cycle outputs, inspect audits, prepare the DB payload, save, and reload.

## Effective Baseline And Walk From

The page distinguishes three layers:

- **Printed**: the source values as stored in the selected baseline.
- **Correction**: an explicit baseline adjustment.
- **Effective**: the baseline state consumed by proposals.

Every requested proposal resolves from its declared Walk From snapshot. A
proposal walking from Requested #1 inherits the effective technical data and
metadata resolved by Requested #1, including applied overrides.

## Mass

Mass owns both the VDE mass and the tire calculation mass contract.

- EPA/TWC resolves a regulatory inertia class.
- GVWR uses curb plus payload as loaded vehicle mass.
- GCWR separates total combination mass from the vehicle load carried by tires.
- Trailer ABC belongs to roadload once; it is not folded into tire mass.

The mass audit exposes VDE mass, tire calculation mass, their bases, status, and
notes for each requested proposal.

## Tire

Tire inputs support inherited values, Tire DB lookup, direct target final RRC,
improvement, and Not Used where valid. Tire DB test load describes the tire
test condition; it is not the vehicle mass used for tire ABC calculation.

The tire resolver uses the canonical mass state published by Mass. Full SAE
lookup therefore changes the tire contribution relative to a compatible source
tire contribution, never by treating a missing source contribution as zero.

## Transmission And Roadload

`ABC_TOTAL` represents total vehicle roadload and `ABC_NET` is the resolved
total less the transmission contribution. Transmission can either retain
measured TOTAL and recompute NET, or apply an explicitly selected vehicle
change to TOTAL.

For transmission coastdown share, the requested share is applied to Walk From
`ABC_TOTAL` coefficient-by-coefficient. TOTAL remains fixed and NET is
recalculated from the estimated transmission contribution.

## Preview, Save, And Reload

Preview exposes validation, engineering comparison, roadload curves, cycle
power analysis, and technical audit. A scenario is saveable only when its
selected proposal is ready. Save is append-only and records the resolved
snapshot, lineage, metadata, provenance, and audit context needed for
historical reload.

## QA And Deferred Work

Use the synthetic QA database under `data/qa/` for deterministic validation;
do not regenerate the local production database as part of QA. The full stable
contract is documented in [VDE Setup v2.2 Final Stable Contract](VDE_SETUP_V22_FINAL_CHECKPOINT.md).

Temperature and ambient-pressure condition scenarios are deferred to
Comparison Report. They should be temporary derived scenarios by default rather
than permanent VDE Setup database rows.

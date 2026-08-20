# EcoDrive — Powertrain Scenario Architecture Guidelines

## Purpose

This document consolidates the main architecture and validation lessons learned during the development of VDE Setup v2.2 and applies them to the future **Powertrain Scenario**, **Comparison Reports**, and longitudinal simulation roadmap.

The guiding principle is:

> The real product is not the UI. It is the set of reproducible physical contracts, resolved scenarios, simulation results, and auditable assumptions that the UI operates.

---

## 1. Reuse the VDE scenario pattern

Recommended flow:

```text
Baseline
→ Corrections / Effective Baseline
→ Scenario Proposal
→ Resolved Scenario
→ Simulation
→ Comparison
→ Save Snapshot
```

The UI should collect inputs, but the core should receive a canonical scenario object.

Conceptually:

```text
UI
→ PowertrainScenario
→ Resolver
→ Simulator
→ ScenarioResult
```

Avoid embedding physical logic or simulation logic directly in Streamlit pages.

---

## 2. Preserve Printed / Correction / Effective semantics

Use the same distinction that proved useful in VDE Setup:

```text
Printed
→ original value from DB/source

Correction
→ explicit engineering correction

Effective
→ value actually used by the model
```

Scenario proposals must operate on the effective source state, not directly on the printed/original value.

This is especially relevant for:

- engine parameters;
- transmission parameters;
- final drive;
- motor parameters;
- battery assumptions;
- efficiency assumptions;
- control assumptions.

---

## 3. Separate source data, assumptions, resolved parameters, and results

Recommended hierarchy:

```text
SOURCE DATA
    ↓
ASSUMPTIONS
    ↓
RESOLVED PARAMETERS
    ↓
SIMULATION
    ↓
RESULTS
```

Example:

```text
Engine max power
→ source/input

Average engine efficiency
→ assumption

Engine speed and torque operating point
→ calculated

Wheel energy
→ calculated from vehicle demand

Fuel energy
→ calculated by powertrain model
```

This separation allows the model to evolve without redesigning the UI.

For example, an early model may use:

```text
engine_efficiency = constant
```

while a later model may use:

```text
engine_efficiency = f(engine_speed, engine_torque)
```

The scenario contract should remain stable across these fidelity upgrades.

---

## 4. Build explicit model-fidelity levels

Do not begin with a complete mini-Simulink.

Recommended progression:

### Level 0 — Energy balance

```text
Wheel Energy
÷
Average Efficiency
=
Fuel / Electrical Energy
```

### Level 1 — Simplified component efficiency

```text
Wheel Power
→ drivetrain efficiency
→ engine/motor power
→ simplified efficiency model
```

### Level 2 — Operating points

```text
Vehicle Speed
→ Gear Selection
→ Engine/Motor Speed
→ Wheel Torque
→ Engine/Motor Torque
→ Efficiency Map Lookup
```

### Level 3 — Hybrid control

```text
Engine
+ Motor
+ Battery
+ Regen
+ SOC
+ Supervisory Control Strategy
```

The simulator should support different fidelity levels through a stable interface rather than separate implementations scattered across the UI.

---

## 5. Keep Vehicle Demand and Powertrain Supply separate

The VDE/Vehicle model answers:

> What does the vehicle require at the wheels?

The Powertrain model answers:

> How does the powertrain supply that demand?

Recommended architecture:

```text
VDE / Vehicle Physics
        ↓
VehicleDemandProfile
        ↓
Powertrain Simulator
        ↓
Fuel / Electrical / Battery Results
```

The Powertrain Scenario should not independently recalculate roadload physics.

---

## 6. Define a formal VehicleDemandProfile contract

A future canonical interface between the VDE Core and Powertrain Simulator should preserve time-domain information.

Suggested fields:

```text
time_s
speed_mps
accel_mps2
roadload_force_N
inertial_force_N
wheel_force_N
wheel_power_W
positive_wheel_power_W
negative_wheel_power_W
distance_m
```

This is preferable to passing only an aggregate VDE value because powertrain operating points depend on instantaneous demand.

Conceptually:

```text
VDE Core
→ VehicleDemandProfile
→ Powertrain Simulator
```

---

## 7. Reuse Walk From semantics as scenario lineage

Powertrain scenarios should support parent-child inheritance.

Possible UI wording:

```text
Based On
Scenario Parent
```

The rule must remain the same as VDE Walk From:

> A child scenario inherits the fully resolved effective state of its selected parent, not the original baseline.

Example:

```text
Baseline
ICE + 8AT

Scenario #1
Based On Baseline
+ MHEV 48V

Scenario #2
Based On Scenario #1
+ Transmission Efficiency Improvement
```

Scenario #2 must start from the resolved MHEV configuration from Scenario #1.

---

## 8. Define neutral and directional invariants before implementing features

Before each capability, write simple physical invariants and automated regressions.

Minimum pattern:

```text
Neutral
→ no change

Better
→ expected improvement direction

Worse
→ expected degradation direction

Inherit
→ does not reapply the parent delta

Based On / Walk From
→ uses the effective parent state
```

Examples for Powertrain Scenario:

```text
Same efficiency
→ same energy consumption

Higher efficiency
→ lower input energy

Lower efficiency
→ higher input energy

Zero motor power
→ hybrid architecture behaves like non-assisted baseline for that contribution

Zero regen capability
→ recovered regen energy = 0

No scenario changes
→ scenario result = baseline result
```

These tests should exist before detailed UI work.

---

## 9. Separate potential regenerative energy from recovered energy

Do not mix the physical opportunity for regeneration with powertrain capability.

Recommended hierarchy:

```text
Negative Wheel Energy
        ↓
Potential Regen Available
        ↓
Regen Capture Limit
        ↓
Recovered Electrical Energy
        ↓
Stored Battery Energy
```

Possible quantities:

```text
E_negative_wheel
E_regen_available
E_regen_captured
E_battery_stored
```

This separation supports transparent HEV/PHEV/BEV modeling.

---

## 10. Comparison Reports must consume resolved results

The Comparison Report must not become a second physics engine.

Correct architecture:

```text
Simulation
→ ScenarioResult
→ Comparison Report
```

Avoid:

```text
Comparison Report
→ recalculates Tire / Roadload / Powertrain physics
```

The report should only compare canonical result snapshots.

---

## 11. Define a canonical ScenarioResult

A future result contract may contain:

```text
VDE_TOTAL
VDE_NET
Rolling Energy
Aero Energy
Positive Inertial Energy
Potential Regen Available
Recovered Regen
Wheel Energy
Transmission Loss
Engine Energy
Motor Energy
Battery Energy
Fuel Energy
Consumption
CO2
```

Not every fidelity level must populate every field.

Unavailable quantities should be explicitly marked rather than silently estimated.

---

## 12. Prepare energy decomposition from the start

Recommended decomposition:

```text
Vehicle Demand
├── Rolling
├── Aero
├── Inertia
└── Other Roadload / Residual

Powertrain
├── Transmission Loss
├── Engine Loss
├── Motor Loss
├── Battery Loss
└── Regen Recovered
```

This enables Comparison Reports to explain *why* a scenario improved rather than only reporting final consumption.

Example:

```text
Rolling Energy       -2.1%
Aero Energy           0.0%
Transmission Loss    -4.2%
Engine Energy        -1.8%
Regen Captured       +8.5%
Final Fuel Energy    -3.7%
```

---

## 13. Preserve provenance, assumptions, and model version

Every saved simulation result should preserve enough information for reproducibility.

Suggested metadata:

```text
Input Snapshot
Effective Scenario Snapshot
Vehicle Demand Source
Cycle / Cycle Version
Model Fidelity Level
Powertrain Model Version
Assumption Set
Efficiency Source
Engine Map Source
Motor Map Source
Battery Model Source
Regen Model Version
Warnings / Limitations
```

Example audit summary:

```text
Powertrain Model: L1
Model Version: 0.3
Engine Efficiency Source: Estimated
Engine Map: Not Available
Transmission Efficiency: Assumed
Regen Model: Simplified
```

A result generated with a simple efficiency assumption must not appear equivalent in fidelity to one generated with real maps.

---

## 14. Keep physics features separate from ML features

The simulation architecture should naturally produce an **Engineering Feature Snapshot**.

Possible fields:

```text
mass
CdA
RRC
ABC_TOTAL
ABC_NET
rolling_energy
aero_energy
positive_inertial_energy
potential_regen
transmission_loss
engine_efficiency
motor_efficiency
battery_capacity
electrification
VDE
fuel_energy
consumption
```

This snapshot can later feed ML, but:

```text
Engineering Features
≠
ML Feature Selection
```

ML pipelines should explicitly choose features and protect against target leakage.

---

## 15. Recommended conceptual objects

Before building the Powertrain Scenario UI, define and validate four central contracts:

```text
PowertrainBaseline
PowertrainScenario
VehicleDemandProfile
ScenarioResult
```

Recommended dependency direction:

```text
PowertrainBaseline
        ↓
PowertrainScenario
        ↓
Resolved Powertrain Scenario
        │
VehicleDemandProfile
        │
        └───────→ Powertrain Simulator
                         ↓
                   ScenarioResult
                         ↓
              Comparison / Persistence
```

---

## 16. Recommended high-level architecture

```text
              VDE / Vehicle Model
                      │
                      ▼
            VehicleDemandProfile
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      VDE Results          Powertrain Scenario
                                  │
                                  ▼
                         Powertrain Simulator
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              Energy Results            Operating Points
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                           ScenarioResult
                                  │
                       ┌──────────┴──────────┐
                       ▼                     ▼
                Comparison Report      Save / History
```

---

## 17. Development order

Recommended order when Powertrain Scenario development begins:

```text
1. Define canonical contracts
2. Define neutral/directional invariants
3. Implement Level 0 simulator
4. Validate energy balance
5. Add deterministic QA scenarios
6. Add Save/Reload reproducibility
7. Build thin UI
8. Build Comparison Report from ScenarioResult
9. Add Level 1 model
10. Add operating points / Level 2
11. Add hybrid control / Level 3 only when justified
```

The UI should come after the core contracts and basic physical invariants are stable.

---

## 18. Core lessons imported from VDE Setup v2.2

1. **UI must not be the physical model.**
2. **Explicit state beats hidden/transient state.**
3. **Neutral scenarios are essential regression tests.**
4. **Directional tests catch physics integration errors early.**
5. **Inheritance must use resolved parent state.**
6. **Preview/result objects should be canonical sources for charts and reports.**
7. **Reports must never independently recalculate physics.**
8. **Save historical snapshots, not references to mutable live data.**
9. **Keep model fidelity and assumptions auditable.**
10. **Complexity must provide real functional value.**

---

## Guiding principle

> First define trustworthy physical contracts and reproducible scenario results. Then build the UI, reports, ML, and agent layers around those contracts.

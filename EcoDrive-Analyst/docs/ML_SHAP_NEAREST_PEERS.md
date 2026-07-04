# ML, SHAP, Nearest Peers, and Regression Notes

## ML Prediction

`ML Prediction` is a runtime inference capability used by `Powertrain Scenario`.

Important boundaries:

- the UI should not execute the full notebook at runtime
- the notebook is an experimental source, not the production inference path
- runtime inference should use an exported artifact
- the runtime result must remain compatible with the common estimation contract
- the current artifact predicts final fuel / energy outputs
- the `PSE` shown in the UI is currently derived from those outputs and the active VDE demand basis
- direct ML prediction of cycle-effective `PSE` is future work unless a dedicated target and artifact are trained for it

### Artifact Expectations

Expected artifact location:

- `models/`

Current repository example:

- `models/powertrain_scenario_ml.joblib`

Optional ML dependencies:

- `requirements-ml.txt`

### Possible Runtime States

The current runtime may report:

- artifact found and loaded
- artifact missing
- artifact load failed
- missing features
- partial / out-of-domain coverage

These are valid status outcomes and should be read as operational states, not necessarily as defects in the UI itself.

## SHAP

### What SHAP Is

SHAP is a model-attribution technique.

In simple terms, SHAP tries to answer:

> Which features influenced the model prediction?

In the current product, that means:

> Which features influenced the model's predicted fuel or energy result?

### What SHAP Is Not

SHAP does not prove physical causality.

SHAP does not prove that a real component is physically correct or physically wrong.

SHAP should be read as a model explainability signal.

Required interpretation warning:

> These are model attribution signals, not proven physical causes.

Additional boundary for the current page:

> SHAP explains the ML result first. Any displayed `PSE` is interpreted afterward from demand and consumed energy.

### Suggested Engineering Grouping

SHAP-style contributions are easier to read when grouped into engineering blocks such as:

- `Roadload / VDE`
- `Mass / Vehicle`
- `Powertrain`
- `Transmission`
- `Brand / model-family residual`

### Simple Example

- `Roadload / VDE` `+0.35 L/100km`
- `Mass / Vehicle` `+0.20 L/100km`
- `Powertrain` `-0.15 L/100km`
- `Transmission` `+0.05 L/100km`

This kind of breakdown is useful for model interpretation, but it still does not prove direct physical root cause.

## Nearest Peers

### Goal

The goal of `Nearest Peers` is to find technically similar vehicles or saved scenarios.

It is not:

- full clustering
- a component-causality engine
- automatic parameter tuning

### Typical Features Used

Nearest-peer comparison can use known technical features such as:

- category
- mass
- engine power
- fuel type
- electrification
- transmission
- `VDE_TOTAL`
- `VDE_NET`
- `ABC`-related values when available through the active request context

### Statistics To Read

Peer-group summaries may include:

- mean
- median
- standard deviation
- min / max
- IQR when available in the current implementation
- delta vs median
- z-score vs peer group
- peer group quality

### Interpretation

- low dispersion means the peer group is relatively coherent
- high dispersion means the comparison is weak or heterogeneous
- a high z-score means the current scenario stands out relative to technically similar peers

### Peer Group Quality

The current quality idea is simple guidance, not certification:

- `High confidence`
  - enough peers and relatively low dispersion
- `Medium confidence`
  - enough peers but more dispersion
- `Low confidence`
  - too few peers or very heterogeneous peers

This is statistical guidance only.

Nearest peers can support the same interpretation ladder:

- demand side: similar `VDE_TOTAL` / `VDE_NET` context
- conversion side: similar efficiency behavior when `PSE` is available
- outcome side: similar final fuel / electric result

## Investigation Hints

Investigation hints are not automatic editors.

They are suggestions for where to look next.

Important rules:

- they do not change inputs automatically
- they do not save physical values automatically
- they are prompts for engineering review

Examples:

- highway penalty -> investigate aero, tires, gearing
- poor fuel result with good VDE -> investigate powertrain efficiency or calibration
- `TOTAL` much higher than `NET` -> review transmission / neutral drag assumptions

With the current page structure, a useful reading sequence is:

1. confirm maneuver and cycle context
2. confirm resolved VDE demand
3. inspect `PSE`
4. inspect final fuel / electric outcome
5. use SHAP and peers to understand why the ML output landed where it did

## Regression

Regression remains an empirical estimation method.

It should be interpreted with care:

- use filters intentionally
- inspect the active scatter / visual context when available
- treat it as an estimation and benchmark tool, not as the center of the page

Regression is valuable for:

- fast empirical estimation
- peer-based comparison
- scenario sanity checks

But it should not be confused with a fully physical explanatory model.

## Comparison Report And Future Benchmark Direction

The current `Comparison Report` is the beginning of a future benchmark/report area.

Future directions include:

- scorecards
- benchmark with technically similar vehicles
- regulatory labels such as:
  - `INMETRO`
  - `CAFE`
  - `UNECE`
  - other planned labels
- performance directions such as:
  - launch
  - gradeability
  - `0-100`
  - `60-100`
  - `80-120`
  - `Vmax`
- future RAG / external technical-data agent support
- hidden component priors as future backlog only

Important note:

- hidden component priors are not a current delivered capability
- nearest peers are not causal inference
- current `PSE` is a cycle-effective interpretation layer, not a direct causal engine state

## Known Limitations

- ML depends on exported artifacts and compatible dependencies
- SHAP depends on explainability support of the active runtime path
- nearest-peer guidance depends on data quality and coverage
- peer guidance is advisory, not authoritative
- regression remains empirical
- current ML explainability applies directly to predicted outputs, not to a dedicated `PSE` target

## Related Docs

- [Sprint 5 Closure](SPRINT_5_CLOSURE.md)
- [Powertrain Scenario Guide](POWERTRAIN_SCENARIO_GUIDE.md)
- [VDE Setup Guide](VDE_SETUP_GUIDE.md)

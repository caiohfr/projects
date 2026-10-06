# EcoDrive Component Estimation — Estimator Contract v1.0

**Status:** FROZEN — Gate 2 passed; holdout validation phase  
**Depends on:** `EcoDrive Component Estimation — Physical Contract v1.0`  
**Purpose:** define the inverse-estimation rules, degrees of freedom, constraints, identifiability, uncertainty, numerical QA and model projection used by Component Estimation v1.

> **Freeze rule:** Gate 2 has passed. The physical and estimator contracts are frozen for the 35-case holdout. Holdout findings may invalidate or motivate a future estimator version, but shall not silently tune v1.0 thresholds, physics, routing, or component boundaries.

---

## 1. Core principle — sampled speed points do not create new information

EPA Target road load:

\[
F_T(v)=F_0+F_1v+F_2v^2
\]

contains **three independent road-load scalars**.

Target + Dyno Set:

\[
F_T(v),\quad F_D(v)
\]

contain at most **six independent road-load scalars** when they are a valid matched pair.

Evaluating those polynomials at 10, 100 or 1000 speeds is useful for numerical fitting and QA, but it **does not increase evidence rank**.

Therefore EcoDrive shall explicitly track:

```text
source_information_rank
free_parameter_count
jacobian_rank
scaled_condition_number
```

and shall not accept a decomposition merely because a many-point least-squares fit converges.

Core rule:

```text
numerical samples != independent evidence
```

---

## 2. Estimator states

```text
E0_FULL_TARGET_SET_ANALYTIC
E1_TARGET_SET_CONSTRAINED
E2_TARGET_ONLY_STRUCTURED
E3_TARGET_ONLY_EVIDENCE_BOUNDED
E4_RESEARCH_BUILDUP
E5_NOT_IDENTIFIABLE
E6_STOP
```

### E0 — Full Target+Set analytic
Use the literature-derived closed-form Moskalik solution when its architecture applicability is valid.

### E1 — Target+Set constrained
Use constrained least squares for architecture-aware models whose parameters are not available through a closed-form solution.

### E2 — Target-only structured
Target ABC are available and the model has enough frozen structural relationships to reduce the free DOF to the available evidence rank.

### E3 — Target-only evidence-bounded
Target ABC plus independent component evidence make an otherwise underidentified problem estimable.

### E4 — Research build-up
Independent component evidence exists but regulatory inverse decomposition is not identifiable. Calculate forward component curves and preserve explicit residual.

### E5 — Not identifiable
The equations, independent constraints and model structure are insufficient to uniquely estimate the requested parameters.

### E6 — Stop
Configuration/test-state identity or required authoritative road-load input is not resolved.

---

## 3. Common quantities

For EPA inputs, convert first to SI:

\[
F_0=A_{lbf}\cdot 4.4482216152605
\]

\[
F_1=B_{lbf/mph}
\frac{4.4482216152605}{1.609344}
\]

\[
F_2=C_{lbf/mph^2}
\frac{4.4482216152605}{1.609344^2}
\]

with speed in km/h.

Default air density for the estimator reference state:

\[
\rho=1.225\;kg/m^3
\]

unless a matched test condition supplies another adopted density.

Aerodynamic force:

\[
F_{aero}(v)=
\frac{\rho CdA}{25.92}v^2
\]

Rolling + Minor baseline form:

\[
F_R(v)=R_0(1+kv)
\]

with:

\[
k=0.00436\;km^{-1}
\]

when the Moskalik fleet-average rolling-speed relationship is invoked.

---

# 4. Conventional multi-speed — Full Target+Set

## 4.1 Canonical solver

```text
architecture = CONVENTIONAL_MULTI_SPEED
state = E0_FULL_TARGET_SET_ANALYTIC
solver = MOSKALIK_2020_ANALYTIC
```

The analytic solution remains canonical. Constrained least squares does **not** replace it.

Let:

\[
L_i=F_i-D_i
\]

Then in SI:

\[
T_2=L_2
\]

\[
T_1=-159T_2
\]

\[
T_0=-36.9T_1+22.2
\]

Vehicle-side rolling remainder:

\[
RT_0=L_0-T_0
\]

\[
RT_1=L_1-T_1
\]

Using:

\[
A_0=-27.5A_1
\]

and the total rolling relation:

\[
R_1=kR_0
\]

solve:

\[
A_1=
\frac{D_1+RT_1-k(D_0+RT_0)}
{1+27.5k}
\]

\[
A_0=-27.5A_1
\]

\[
RD_0=D_0-A_0
\]

\[
RD_1=D_1-A_1
\]

\[
R_0=RD_0+RT_0
\]

\[
R_1=RD_1+RT_1
\]

The effective pure-quadratic aero coefficient is:

\[
C_{aero}=D_2+\frac{A_1}{133}
\]

and:

\[
CdA=C_{aero}\frac{25.92}{\rho}
\]

This branch is the regression oracle for the generalized estimator.

---

# 5. Conventional multi-speed — Target-only structured estimator

A major v0.1 decision is to permit a **structured Target-only estimate** even when no independent CdA/RRC source exists.

This follows the EcoDrive principle:

```text
calculate-first, provenance-always
```

but it can never receive `SUPPORTED` without stronger evidence.

## 5.1 Structural assumptions

Use:

\[
R_1=kR_0
\]

\[
T_2=-T_1/159
\]

\[
T_0=-36.9T_1+22.2
\]

and pure-quadratic aero.

Then:

\[
F_0=R_0+T_0
\]

\[
F_1=kR_0+T_1
\]

\[
F_2=\frac{\rho CdA}{25.92}+T_2
\]

The free parameter vector is:

\[
\theta=[R_0,T_1,CdA]
\]

Exactly three independent Target coefficients constrain exactly three free parameters.

## 5.2 Solver

```text
state = E2_TARGET_ONLY_STRUCTURED
solver = BOUNDED_LINEAR_LS_MOSKALIK_TARGET_ONLY
source_information_rank = 3
free_parameter_count = 3
```

Although an analytic solution exists, use a bounded linear least-squares implementation so independent bounds can be added without changing the solver interface.

Hard physics constraints:

```text
R0 >= 0
CdA > 0
```

Do **not** constrain the signs of `T0/T1/T2` individually.

Instead require:

```text
F_trans(v) >= 0
```

over the applicable physical QA domain.

## 5.3 Promotion

With no independent component evidence:

```text
maximum status = CONDITIONAL
reason = STRUCTURAL_FLEET_ASSUMPTIONS_ONLY
```

With matched independent CdA and/or rolling evidence, apply those constraints and retain their provenance.

---

# 6. External evidence constraint policy

Evidence is not all treated the same.

## 6.1 Fixed / hard constraint

May be used when:

```text
configuration_match = EXACT
and
evidence is SOURCE/MEASURED or vehicle-specific RESEARCHED
and
the measurement boundary is compatible
```

Examples:

```text
published CdA with exact vehicle configuration
measured component force curve
matched axle weight distribution
explicit engineering bound with known test condition
```

## 6.2 Interval constraint

If the source gives a defensible range:

```text
lower <= parameter <= upper
```

Do not collapse the range to a nominal value before fitting.

## 6.3 Validation-only evidence

Use for:

```text
RESEARCHED_APPROX
same-generation transfer
body-category Cd
tire RRC under non-matching conditions
generic component benchmark
```

These do not silently pull the optimizer.

They are used after fitting for QA / plausibility.

## 6.4 Tire RRC rule

Tire-only RRC is not equal to EcoDrive `Rolling+Minor`.

Therefore:

```text
RRC_tire != hard equality constraint on RRC_eq
```

When test conditions are sufficiently compatible, tire RRC may act as a **physical floor / plausibility check**, because Rolling+Minor also contains unresolved minor mechanical losses.

---

# 7. EV/FCEV fixed gear — Target+Set

## 7.1 Primary model

\[
F_{EDU}(v)=K_0+K_bv^{2/3}+K_cv^2
\]

with:

\[
K_0,K_b,K_c\ge0
\]

The Target/Set boundary is used only when the dynamometer configuration is physically interpretable.

For a 2WD vehicle define:

```text
q_roll = fraction of Rolling+Minor remaining on the vehicle side
```

Then:

\[
F_D(v)
=
F_{aero}(v)
+
(1-q_{roll})F_R(v)
\]

\[
F_L(v)
=
F_T(v)-F_D(v)
=
F_{EDU}(v)
+
q_{roll}F_R(v)
\]

with:

\[
F_R(v)=R_0(1+kv)
\]

Free parameters:

\[
\theta=[CdA,R_0,K_0,K_b,K_c]
\]

Once `q_roll` and `k` are frozen, there are five free parameters against six independent Target+Set coefficients.

## 7.2 q_roll hierarchy

Use in order:

```text
1. measured / published driven-axle static load fraction
2. matched engineering weight-distribution evidence
3. approved assumption q_roll = 0.50
```

The 0.50 fallback is an EcoDrive v1 engineering assumption, not a literature claim.

If the 0.50 assumption is used:

```text
status <= CONDITIONAL
```

and run a mandatory sensitivity check at:

```text
q_roll = 0.40
q_roll = 0.50
q_roll = 0.60
```

This sweep is a robustness test, not a probability distribution.

## 7.3 AWD / 4WD

For AWD/4WD fixed-gear vehicles:

```text
Target+Set primary inverse split = NOT_IDENTIFIABLE_V1
```

unless the dyno/test boundary and rolling allocation are independently resolved.

Use:

```text
MOSKALIK_FALLBACK_EXPERIMENTAL
```

or `RESEARCH_BUILDUP`.

## 7.4 Identifiability caution

`K0`, `Kb`, `Kc` shall not all be estimated from Target-only ABC while CdA and rolling remain free.

Target-only ABC has rank 3; the model would have more free physical parameters than independent road-load evidence.

---

# 8. EV/FCEV fixed gear — Target-only

Primary fixed-gear estimation is allowed only if enough independent evidence freezes the missing component degrees of freedom.

Typical identifiable case:

```text
Target ABC
+ independently constrained CdA
+ independently constrained Rolling+Minor
→ fit K0, Kb, Kc
```

If only Target ABC are available:

```text
FIXED_GEAR_EDRIVE_V1 = NOT_IDENTIFIABLE
MOSKALIK_FALLBACK_EXPERIMENTAL = may be attempted
```

No generic regularization may be used to convert an underidentified five-parameter split into a promoted physical result.

---

# 9. DHT / PowerSplit / Series-Parallel — aggregated primary

The v1 DHT model remains aggregate.

For a 2WD Target+Set case with a resolved/frozen `q_roll`:

\[
F_D(v)
=
F_{aero}(v)+(1-q_{roll})F_R(v)
\]

Fit only:

\[
\theta=[CdA,R_0]
\]

to the Dyno Set curve.

Then compute:

\[
F_{DHT,agg}(v)
=
[F_T(v)-F_D(v)]
-
q_{roll}F_R(v)
\]

No internal planetary / MG1 / MG2 / clutch decomposition is attempted.

If `q_roll=0.50` is assumed:

```text
maximum status = CONDITIONAL
```

For Target-only DHT:

```text
independent CdA + Rolling evidence available
    → aggregate drivetrain residual may be calculated

otherwise
    → primary = NOT_IDENTIFIABLE
    → Moskalik fallback may be attempted
```

---

# 10. EV multi-speed

```text
primary_model = NONE_V1
```

Therefore:

```text
Target+Set or Target-only
→ MOSKALIK_FALLBACK_EXPERIMENTAL where mathematically possible
→ maximum status CONDITIONAL
```

No least-squares parameterization of a fictional universal multi-speed EV loss curve is introduced in v1.

---

# 11. Least-squares objective

Least squares is used as a **numerical estimator**, not as a source of physical knowledge.

For a model curve:

\[
\hat F(v,\theta)
\]

use:

\[
J_{road}
=
\frac{1}{V_{max}-V_{min}}
\int_{V_{min}}^{V_{max}}
\left[
F_{obs}(v)-\hat F(v,\theta)
\right]^2dv
\]

implemented by a deterministic uniform quadrature grid.

For Target+Set models:

\[
J=J_D+J_L
\]

or the equivalent simultaneous stacked residual system.

Important:

```text
quadrature grid density changes numerical integration accuracy
but does not change source_information_rank
```

Do not introduce generic L1/L2 regularization in v1 merely to force uniqueness.

If a solution needs arbitrary regularization to exist:

```text
status = NOT_IDENTIFIABLE
```

---

# 12. Identifiability gate

Run before promotion.

## 12.1 Structural gate

Record:

```text
source_information_rank
free_parameter_count
frozen_parameter_count
structural_equality_count
```

If the requested model has more unresolved physical degrees of freedom than independent evidence can support:

```text
NOT_IDENTIFIABLE
```

## 12.2 Jacobian / design-matrix rank

Using dimensionless column scaling:

```text
rank(J_scaled) == free_parameter_count
```

must hold.

Exact rank deficiency:

```text
NOT_IDENTIFIABLE
```

## 12.3 Condition number

EcoDrive v1 numerical thresholds:

```text
scaled condition number < 100
    → numerically acceptable

100 <= condition number <= 1000
    → CONDITIONAL / ILL_CONDITIONED

condition number > 1000
    → NOT_IDENTIFIABLE
```

These are EcoDrive engineering QA thresholds, not regulatory thresholds.

---

# 13. Hard physical constraints

Common:

```text
finite parameters
CdA > 0
R0 >= 0
```

Fixed-gear native model:

```text
K0 >= 0
Kb >= 0
Kc >= 0
```

Moskalik:

```text
do not require T0/T1/T2 individually positive
require F_trans(v) materially non-negative in the hard QA domain
```

Rolling:

```text
require F_rolling+minor(v) materially non-negative
```

Aero:

```text
require F_aero(v) >= 0
```

A signed **model residual** is allowed as a diagnostic.

A signed residual shall not be renamed into a physical loss component.

## 13.1 Active-bound / component-collapse diagnostic

A constrained optimizer reaching a legal mathematical bound is not automatically a physical success.

Record for every free parameter:

```text
at_lower_bound
at_upper_bound
distance_to_bound
```

If a physically necessary aggregate component collapses to approximately zero solely because the optimizer is constrained there, flag:

```text
BOUND_ACTIVE_PHYSICAL_DEGENERACY
```

For example, whole-vehicle `Rolling + Minor` cannot be interpreted as a physically valid zero-loss component simply because `R0 >= 0` was satisfied numerically.

Rules:

```text
active bound + physically plausible solution
    → warning only

active bound + strong sensitivity to an assumption
    → maximum status CONDITIONAL

physically necessary component collapses to zero
and external/structural evidence contradicts that result
    → REJECTED_PRIMARY_MODEL
    → attempt approved fallback
```

This diagnostic is separate from closure and condition number.

---

# 14. Quantitative reconstruction QA

Primary metric:

\[
NRMSE=
\frac{RMSE}
{\operatorname{mean}(|F_{Target}|)}
\]

over the model QA domain.

EcoDrive v1 candidate thresholds:

```text
NRMSE <= 2.0%
    → closure PASS

2.0% < NRMSE <= 5.0%
    → closure CONDITIONAL

NRMSE > 5.0%
    → REJECTED_MODEL
```

Secondary metric:

\[
E_{max,norm}=
\frac{\max|F_T-\hat F_T|}
{\operatorname{mean}(|F_T|)}
\]

Candidate thresholds:

```text
<= 5%
    → PASS

5–10%
    → CONDITIONAL

> 10%
    → REJECTED_MODEL
```

These thresholds are versioned EcoDrive criteria and must be recalibrated only on the development set, never silently changed during holdout validation.

---

# 15. Sensitivity / stability QA

v1 does not assign Bayesian probabilities to assumptions for which no probability distribution is known.

Use deterministic envelopes.

Perturb only declared uncertain inputs / assumptions, for example:

```text
CdA source range endpoints
RRC evidence range endpoints
q_roll = 0.40 / 0.50 / 0.60 when 0.50 is assumed
alternate approved structural case
```

For each output parameter/reporting KPI:

\[
S_{rel}
=
\frac{max-min}
{\max(|nominal|,\epsilon)}
\]

Candidate interpretation:

```text
S_rel <= 10%
    → STABLE

10% < S_rel <= 25%
    → SENSITIVE

S_rel > 25%
    → HIGHLY_SENSITIVE
```

`HIGHLY_SENSITIVE` prevents `SUPPORTED` promotion.

No random Monte Carlo distribution is implied by this envelope.

---

# 16. Rolling-model structural sensitivity

Baseline:

\[
R(v)=R_0(1+0.00436v)
\]

The Moskalik paper explicitly notes that rolling allocation is more sensitive than aero to alternate rolling-speed assumptions.

Therefore a future approved alternate rolling form may be used as an **ablation / structural sensitivity case**.

It shall not be mixed into the nominal result silently.

---

# 17. Native fixed-gear model → equivalent ABC

The native model remains authoritative:

\[
F_{EDU}=K_0+K_bv^{2/3}+K_cv^2
\]

For interoperability with EcoDrive component-ABC interfaces, fit:

\[
F_{EDU}(v)
\approx
A_{eq}+B_{eq}v+C_{eq}v^2
\]

using ordinary least squares over:

```text
0–130 km/h
```

or the intersection with the declared native-model validity range.

Use uniform speed weighting so the projection is not tied to one regulatory cycle.

Store:

```text
A_eq
B_eq
C_eq
projection_speed_min
projection_speed_max
projection_RMSE
projection_NRMSE
projection_max_abs_error
```

The equivalent ABC is a compatibility representation, not the physical law.

---

# 18. Residual policy

Always distinguish:

```text
component_estimate
model_residual
unresolved_physical_component
```

Do not force:

```text
Aero + Rolling + Drivetrain = Target
```

by inventing an unnamed physical component.

Instead:

\[
Residual(v)=F_{Target}(v)
-
[
F_{aero}(v)+F_R(v)+F_{drivetrain}(v)
]
\]

A small signed residual may be retained as numerical/model discrepancy.

A substantial residual means:

```text
CONDITIONAL
or
REJECTED_MODEL
```

depending on QA thresholds.

---

# 19. Required estimator output contract

```text
estimator_method
estimator_version
estimator_state

configuration_match_status
architecture_class
model_type
model_role

source_information_rank
free_parameter_count
frozen_parameters
free_parameters
active_constraints
jacobian_rank
scaled_condition_number

fit_speed_min
fit_speed_max

CdA
CdA_provenance
frontal_area_m2
Cd_implied

R0
R1
RRC_eq_0
RRC_eq_80

drivetrain_native_model
drivetrain_native_parameters
drivetrain_eq_A
drivetrain_eq_B
drivetrain_eq_C

target_RMSE
target_NRMSE
target_max_abs_error
target_max_abs_error_norm

set_RMSE
vehicle_loss_RMSE

model_residual_metrics

sensitivity_cases
parameter_envelopes
stability_status

hard_fail_flags
soft_warning_flags
reason_codes

final_component_status
```

---

# 20. Status promotion logic

```text
STOP identity?
    → STOP_*

otherwise
    ↓
model structurally identifiable?
    no → NOT_IDENTIFIABLE
    yes
    ↓
solver converged?
    no → REJECTED_MODEL
    yes
    ↓
hard physical QA pass?
    no → REJECTED_MODEL
    yes
    ↓
closure pass?
    no → REJECTED_MODEL / CONDITIONAL
    yes
    ↓
model in primary architecture domain?
    no → max CONDITIONAL
    yes
    ↓
evidence sufficient + stability acceptable?
    yes → SUPPORTED
    no  → CONDITIONAL
```

---

# 21. Initial regression evidence

## 21.1 Carnival — Target-only structured vs full Target+Set

Using the existing exact Carnival EPA pilot:

Full Target+Set Moskalik result:

```text
CdA = 0.96163 m²
RRC_eq_80 = 7.916 N/kN
```

Using **Target only** with the structured three-DOF model:

```text
CdA = 0.96359 m²
RRC_eq_80 = 7.886 N/kN
```

Difference in CdA is approximately:

```text
+0.20%
```

This is strong evidence that the Target-only structured branch is worth retaining as a conditional estimator.

It does not prove universal validity.

## 21.2 CLA35 — Target-only structured

The same no-external-bound Target-only structural model gives approximately:

```text
CdA = 0.6504 m²
RRC_eq_0 = 9.32 N/kN
RRC_eq_80 = 12.58 N/kN
```

The earlier tire-evidence-bounded pilot produced a CdA interval around:

```text
0.654–0.666 m²
```

Interpretation:

```text
the structural solution is usable as a conditional baseline,
but independent evidence materially improves allocation.
```

## 21.3 Fixed-gear caution

A key Gate-2 correction:

A three-parameter fixed-gear drivetrain model plus free aero and rolling **cannot be claimed uniquely identified from Target-only ABC**.

Any prior fixed-gear result that did not explicitly record sufficient independent evidence / frozen allocation assumptions must be treated as:

```text
MODEL_PROJECTION / CONDITIONAL
```

until rerun under this estimator contract.

This does not invalidate the fixed-gear physical model. It corrects the estimator-identifiability claim.

---

# 22. Development / validation protocol

Use the existing 50-vehicle cross-OEM sample.

Before tuning thresholds, freeze:

```text
15 development/regression cases
35 holdout validation cases
```

The 15-case set must span:

```text
conventional Target+Set
conventional Target-only
fixed gear EV/FCEV
parallel hybrid conventional transmission
power-split/DHT
fallback Moskalik
matching STOP
not-identifiable case
```

STOP/matching cases remain regression tests even when they have no numeric component result.

Rules:

```text
thresholds may be calibrated only on development cases
holdout is not used to tune
if holdout forces methodology changes:
    bump estimator version
    record that holdout was contaminated
```

---

# 22.1 Early Gate-2 development findings

The first explicit constrained runs added two estimator QA lessons.

### Mirai fixed-gear

With `q_roll = 0.50`, the primary fixed-gear model closes the Target curve with approximately **1.72% NRMSE**, which is now a closure `PASS` under v0.2.

However:

- the native drivetrain coefficients collapse to the lower bound at the nominal split;
- changing `q_roll` to 0.40 produces a materially different non-zero drivetrain solution;
- changing `q_roll` to 0.60 pushes Target NRMSE slightly above 5%.

Therefore the final component status remains:

```text
CONDITIONAL
reason_codes:
    ASSUMED_Q_ROLL
    ACTIVE_BOUND_DRIVETRAIN
    HIGH_Q_ROLL_SENSITIVITY
```

### Mustang Mach-E fixed-gear

Across `q_roll = 0.40 / 0.50 / 0.60` the current primary fixed-gear Target+Set formulation returns approximately **4.22% Target NRMSE**, which is closure `CONDITIONAL`.

However the rolling term collapses essentially to zero for all three assumptions.

Therefore:

```text
FIXED_GEAR_EDRIVE_V1 primary
    → REJECTED_PRIMARY_MODEL
    → reason = ROLLING_COLLAPSED_TO_ZERO

MOSKALIK_FALLBACK_EXPERIMENTAL
    → allowed
    → maximum status CONDITIONAL
```

This is a methodological success of the new QA contract: acceptable closure alone does not override an implausible component allocation.

---

# 23. Gate 2 exit criteria

Estimator Contract may be frozen as v1.0 after:

```text
[ ] reference implementation matches analytic Carnival full Target+Set
[ ] Target-only structured Carnival regression passes
[ ] CLA35 bounded/structured behavior is reproduced
[ ] at least 2 fixed-gear cases rerun with explicit q_roll/evidence state
[ ] at least 1 DHT/PowerSplit aggregate case runs
[ ] NOT_IDENTIFIABLE test is demonstrated
[ ] STOP matching/state cases remain STOP
[ ] 10–15 development cases pass expected behavior
[ ] thresholds are frozen
```

Only then:

```text
Estimator Contract v1.0 = FROZEN
```

and the 35-case holdout / 50-case reporting phase begins.

---

# 24. Implementation ownership

Recommended dependency direction:

```text
source evidence
    ↓
component estimation input contract
    ↓
architecture router
    ↓
identifiability gate
    ↓
analytic / constrained solver
    ↓
physical QA
    ↓
uncertainty envelope
    ↓
canonical ComponentEstimateResult
    ↓
DB / reports / UI
```

No Streamlit-specific estimator physics.

No ML in the deterministic v1 estimator.

No RAG/agent-generated number may bypass the evidence/provenance contract.

---

# 25. Reference-prototype observations

The reference implementation was executed against the currently available pilot inputs.

## 25.1 Carnival regression

The prototype reproduces the existing full Target+Set analytic result:

```text
CdA = 0.9616307651 m²
R0 = 137.0601299 N
R1 = 0.5975821663 N/kph
```

The Target-only structured branch returns:

```text
CdA = 0.9635889152 m²
RRC_eq_80 = 7.8863 N/kN
scaled condition number = 63.6
```

This lies inside the current numerical-conditioning acceptance band and remains close to the full Target+Set result.

## 25.2 Mirai fixed-gear q_roll sensitivity

Using the Mirai Target+Set state currently used by the Gate-1 numeric check, the 2WD fixed-gear constrained estimator produced:

| q_roll | CdA [m²] | Target NRMSE | Interpretation |
|---:|---:|---:|---|
| 0.40 | 0.6672 | 0.31% | closure PASS |
| 0.50 | 0.6963 | 1.72% | closure PASS |
| 0.60 | 0.7811 | 5.13% | REJECTED by candidate closure threshold |

The normalized design-matrix condition number remains around 42–44, so the issue is **not numerical matrix singularity**. It is physical sensitivity to the frozen rolling-allocation assumption.

At q_roll >= 0.50, the non-negative optimizer also drives the native drivetrain parameters toward their lower bounds, another warning that the assumed split is dominating the physical result.

Therefore:

```text
q_roll assumed rather than evidenced
→ fixed-gear primary cannot be SUPPORTED
→ mandatory sensitivity envelope
```

This is a direct example of why convergence + good conditioning + one nominal closure result are not enough for physical promotion.


---

# 26. Gate-2 freeze record — v1.0

Gate 2 closed using reproducible EPA 2026 Test Car source rows plus the existing exact pilot cases. No physical equation was changed to obtain closure.

Frozen interpretation rules:

```text
closure != physical identifiability
source row + configuration/test-state identity are part of regression evidence
historical numeric evidence without reproducible source inputs is not an oracle
multiple valid test articles are preserved as distinct states, not merged
holdout may falsify v1.0 but may not tune v1.0
```

New exact development evidence includes:

```text
Mustang GTD        conventional exact rerun
Buick Enclave FWD  conventional exact rerun
Kia Sorento HEV    conventional-transmission hybrid exact rerun
Hyundai NEXO       two distinct fixed-gear FCEV test articles
Ford Escape HEV    DHT aggregate exact rerun
Cadillac LYRIQ-V   AWD fixed-gear NOT_IDENTIFIABLE gate
```

The NEXO case resolves the former matching STOP by retaining both valid EPA test articles independently. The LYRIQ-V case demonstrates the distinction between missing evidence (`STOP`) and a fully specified road-load state whose requested internal decomposition is structurally underidentified (`NOT_IDENTIFIABLE`).

The development-set thresholds in Sections 12–15 are now frozen. Any holdout-driven methodology change requires a new estimator version and a contamination note.

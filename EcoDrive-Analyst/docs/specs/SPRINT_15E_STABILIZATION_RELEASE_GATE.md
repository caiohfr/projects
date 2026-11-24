# Sprint 15E stabilization and release gate

Status: `IMPLEMENTED — RELEASE GATE PENDING`

Date: 2026-11-24

Verified base: `origin/main@fdf33edb5a1b4ee36578fbb29201b105fa766795`

## Scope

Sprint 15E stabilizes the Sprint 15 presentation work without changing the
Vehicle Demand physics, schemas, or product capability set. It closes two P0
integrity gaps in Database Management, aligns the visible Vehicle Demanded
Energy equation with the canonical core, and records the remaining release
evidence honestly.

## Database stage identity

The session stage is bound, at the first staged operation, to the canonical
absolute path returned by `current_db_path()`. Path comparison resolves
relative components and normalizes case. The stage retains this identity until
it is empty or is explicitly discarded.

The identity is checked:

1. before a new operation or spreadsheet row is added;
2. before review previews are built;
3. before every operation in a commit sequence.

When the active database differs, no preview, stage mixing, receipt lookup, or
write is allowed. Returning to the original database restores the untouched
stage. Discard is an explicit, confirmed action and never writes to either
database.

The application still uses the process-global `src.vde_core.db.DB_PATH` for
runtime database selection. This patch prevents a session stage from crossing
that boundary, but it does not redesign process-global database selection for
multiple concurrent sessions. That architectural limitation remains explicit.

## Commit semantics

Decision: `SEQUENTIAL_WITH_PARTIAL_STATUS`.

The existing persistence services own independent per-operation transactions.
Wrapping those services in a second UI transaction would duplicate transaction
ownership and risk divergent persistence behavior. The coordinator therefore
uses the existing services sequentially and makes partial completion explicit:

- every staged command receives a stable `stage_operation_id`;
- that ID becomes the canonical change-log `operation_id`;
- a successful operation is removed from the stage only after the service
  returns `committed=True` or a matching receipt is found;
- on the first unconfirmed exception, that operation and all later operations
  remain pending and no later operation is attempted;
- retry checks the receipt before applying, preventing replay of a confirmed
  create or spreadsheet row;
- a result without confirmed commit status blocks automatic retry and requires
  reconciliation;
- SQLite and unexpected exceptions are retained in the technical log and
  summarized precisely in the UI.

The last commit receipt table and spreadsheet report distinguish confirmed,
pending, and recovered operations. Discarding a partial stage does not undo
already committed operations.

## Methodology parity

The Methodology & Models page now displays the same positive-power integration
used by `src/vde_core/vde_calc.py`:

```text
VDE = (1 / d_cycle) integral from 0 to T of max(0, P_tractive(t)) dt
```

The page states the SI chain (N, m/s, W, s, km, then MJ/km) and the boundary:
negative tractive power is braking demand and receives no regenerative-energy
credit in VDE. No core formula changed. The UI test evaluates a cycle containing
negative tractive power against `compute_vde_net` and its canonical clipped
series.

Canonical references:

- `docs/architecture/vehicle_demand_architecture.md`
- `docs/sprints/PACKAGE_7G_VDE_NET_TOTAL_CONTRACT.md`
- `docs/sprints/PACKAGE_9B_VEHICLE_DEMAND_ENGINE.md`
- `src/vde_core/vde_calc.py`

## Verification contract

The automated release evidence includes named STG-01 through STG-08 and CMT-01
through CMT-08 scenarios, the Methodology parity regression, affected historical
tests, `git diff --check`, and a final full-suite comparison with the verified
baseline.

The real-browser G4 remains separate from automated AppTest evidence. Sprint 15
may be marked closed only after all eight pages are reviewed at desktop and
narrow widths and the owner explicitly approves the result.

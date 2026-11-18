# Sprint 15A visual parity evidence

Date: 2026-11-18

Baseline: `d0703449` (PR #18 merge)

Working branch: `fix/sprint-15a-visual-parity`

## Scope delivered

- Replaced the duplicated auto-discovered/custom menu combination with one
  `st.navigation` registry containing the eight real destinations.
- Added a shared Streamlit shell with the approved v0.2 navy sidebar, active
  navigation treatment, white contextual topbar and workspace background.
- Restyled only the Home surface. Existing VDE Setup, Load Case Lab,
  Powertrain, Comparison, Database, Methodology and Legacy workflows were not
  redesigned or reimplemented.
- Restored the approved dark hero composition and reused the supplied v0.2
  vehicle image as a local application asset.
- Preserved the PR #18 empty/active Home states, validated `home_context`,
  explicit draft discard and persisted-baseline-only Lab handoff.
- Added a direct AppTest assertion that all seven non-Home destinations are
  registered by the shared navigation. Home is the eighth and default page.

No file under `src/vde_core`, database schema, migration, physical formula,
TOTAL/NET rule, FuelCons calculation or persistence owner changed.

## Token mapping

| v0.2 token | Streamlit owner | Applied value |
| --- | --- | --- |
| Navy | `src/vde_app/visual_theme.py` | `#0a2b46` |
| Sidebar navy | `src/vde_app/visual_theme.py` | `#082940` |
| Action blue | `src/vde_app/visual_theme.py` | `#1769a6` |
| Positive green | `src/vde_app/visual_theme.py` | `#169f73` |
| Workspace background | `src/vde_app/visual_theme.py` | `#f4f7fb` |
| Border | `src/vde_app/visual_theme.py` | `#dce5ef` |
| Primary ink | `src/vde_app/visual_theme.py` | `#0b2742` |
| Muted text | `src/vde_app/visual_theme.py` | `#6d8298` |

Home-specific geometry remains local to `pages/home_page.py`; global tokens
and shell selectors have one owner.

## Fresh verification

- `python -m unittest tests.test_home_context tests.test_home_page_15a -v`:
  **8 passed** after adding the navigation registration test.
- Directed Home and affected-destination set:
  **158 run; 154 passed; 3 failed; 1 error**.
- The four non-passing tests reproduce the same checkout fixture gaps recorded
  for PR #18: the canonical test databases do not contain `tire_db` for two VDE
  Setup lookup cases or `fuelcons_db` for two Comparison database-switch cases.
  None of the failing stacks enters the files changed by this visual patch.
- `py_compile` passed for `app.py`, `pages/home_page.py` and
  `src/vde_app/visual_theme.py`.

## Browser evidence

The Streamlit server started successfully at `http://localhost:8502`.
Interactive capture was attempted through both available routes, but the
integrated browser returned `browser access host is unavailable` and Windows
Computer Use returned `native pipe is unavailable`. No screenshot or
two-viewport visual pass is claimed.

Status: **visual gate pending owner review in a real browser**. This change is
ready for review but does not by itself close the 15A visual parity gate.

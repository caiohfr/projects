from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import sqlite3
import shutil
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.vde_app.components import vde_request_compact
from src.vde_app.components.vde_request_compact import V22_SESSION_KEY, render_active_v22_section
from src.vde_app.components.vde_request_compact_units import to_canonical_field_value
from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import QA_DATA_DIR, seed_qa_database
from src.vde_core.vde_request_compact_state import (
    apply_v22_baseline,
    apply_v22_domain_inputs,
    apply_v22_new_test_baseline,
    apply_v22_proposal_matrix,
    build_v22_canonical_request_draft,
    create_v22_state,
)
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_PATH = ROOT / "src" / "vde_app" / "components" / "vde_request_compact.py"
STYLE_PATH = ROOT / "src" / "vde_app" / "components" / "vde_request_compact_style.py"
VIEWMODEL_PATH = ROOT / "src" / "vde_app" / "components" / "vde_request_compact_viewmodels.py"
PAGE_PATH = ROOT / "pages" / "VDE_Setup.py"
V21_PAGE_PATH = ROOT / "docs" / "archive" / "pages" / "VDE_Setup_v2_1_legacy.py"


def _baseline_row() -> dict:
    return {
        "id": 4998,
        "make": "AUDI",
        "model": "Q6",
        "year": 2027,
        "legislation": "EPA",
        "cycle_name": "FTP75",
        "description": "Baseline request row",
        "mass_kg": 1600.0,
        "test_mass_kg": 1736.0,
        "inertia_class": 1750.0,
        "weight_dist_fr_pct": 55.0,
        "cda_m2": 0.62,
        "A": 120.0,
        "B": 0.02,
        "C": 0.01,
        "rrc_N_per_kN": 8.4,
        "front_pressure_psi": 35.0,
        "rear_pressure_psi": 35.0,
        "brake_A_coef_N": 4.0,
        "brake_B_Npkph": 0.001,
        "brake_C_coef_Npkph2": 0.0001,
    }


def _candidate_rows() -> list[dict]:
    return [
        {
            "VDE ID": 4998,
            "Make": "AUDI",
            "Model": "Q6",
            "Year": 2027,
            "Legislation": "EPA",
            "Cycle": "FTP75",
            "Test mass": 1736.0,
            "ABC_TOTAL": "120 / 0.02 / 0.01",
            "VDE_TOTAL": 1.234,
            "Notes": "Baseline request row",
        },
        {
            "VDE ID": 5001,
            "Make": "VOLVO",
            "Model": "XC40",
            "Year": 2028,
            "Legislation": "EPA",
            "Cycle": "FTP75",
            "Test mass": 1800.0,
            "ABC_TOTAL": "121 / 0.03 / 0.01",
            "VDE_TOTAL": 1.345,
            "Notes": "Candidate row",
        },
    ]


class TestVdeSetupV22Helpers(unittest.TestCase):
    def test_scenario_metadata_editor_groups_are_disjoint(self):
        self.assertFalse(
            set(vde_request_compact.METADATA_SIMPLE_FIELDS)
            & set(vde_request_compact.METADATA_ADDITIONAL_FIELDS)
        )

    def test_metadata_editor_uses_effective_values_and_clears_inherited_override(self):
        source = {"make": "AUDI", "model": "Q5", "model_year": 2026}
        effective = dict(source)

        unchanged = vde_request_compact._normalize_metadata_editor_overrides(
            {"make": "AUDI", "model": "Q5", "model_year": "2026"},
            source_metadata=source,
            effective_metadata=effective,
            existing_overrides={},
            metadata_source="inherit",
        )
        changed = vde_request_compact._normalize_metadata_editor_overrides(
            {"make": "BMW", "model": "Q5", "model_year": "2026"},
            source_metadata=source,
            effective_metadata=effective,
            existing_overrides={},
            metadata_source="inherit",
        )
        reverted = vde_request_compact._normalize_metadata_editor_overrides(
            {"make": "AUDI"},
            source_metadata=source,
            effective_metadata={**effective, "make": "BMW"},
            existing_overrides={"make": "BMW"},
            metadata_source="inherit",
        )

        self.assertEqual(unchanged, {})
        self.assertEqual(changed, {"make": "BMW"})
        self.assertEqual(reverted, {})

    def test_copied_metadata_keeps_effective_values_and_copied_provenance(self):
        copied = vde_request_compact._normalize_metadata_editor_overrides(
            {"make": "VOLVO", "model": "XC40", "model_year": "2028"},
            source_metadata={"make": "AUDI", "model": "Q5", "model_year": 2026},
            effective_metadata={"make": "VOLVO", "model": "XC40", "model_year": 2028},
            existing_overrides={"make": "VOLVO", "model": "XC40", "model_year": 2028},
            metadata_source="existing_vde",
        )

        self.assertEqual(copied["make"], "VOLVO")
        self.assertEqual(copied["model"], "XC40")
        self.assertEqual(copied["model_year"], "2028")
        self.assertEqual(
            vde_request_compact._metadata_provenance("make", copied, {"metadata_source": "existing_vde"}),
            "copied",
        )

    def _app(self) -> AppTest:
        return AppTest.from_file(str(PAGE_PATH))

    def _temp_db_path(self) -> Path:
        QA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="v22_db_", dir=str(QA_DATA_DIR)))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir / "qa_seed.db"

    def _seed_baseline_browser_db(self, db_path: Path, *, vde_id: int, make: str, model: str) -> None:
        previous_path = db_module.current_db_path()
        try:
            db_module.configure_db_path(db_path)
            db_module.ensure_db()
        finally:
            db_module.configure_db_path(previous_path)
        with sqlite3.connect(str(db_path)) as con:
            con.execute("DELETE FROM vde_db;")
            con.execute(
                """
                INSERT INTO vde_db (
                    id,
                    legislation,
                    category,
                    make,
                    model,
                    year,
                    notes,
                    mass_kg,
                    test_mass_kg,
                    inertia_class,
                    coast_A_N,
                    coast_B_N_per_kph,
                    coast_C_N_per_kph2,
                    cycle_name,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (
                    int(vde_id),
                    "EPA",
                    "QA",
                    make,
                    model,
                    2026,
                    f"{model} seeded for baseline browser switching",
                    1500.0,
                    1644.0,
                    1644.0,
                    120.0,
                    0.02,
                    0.01,
                    "FTP75",
                ),
            )
            con.commit()

    def _loaded_state(self) -> dict:
        return apply_v22_baseline(create_v22_state(), _baseline_row())

    def _run(self, app: AppTest, *, timeout: int = 60) -> None:
        app.run(timeout=timeout)

    def _button(self, buttons, *, label: str | None = None, key: str | None = None):
        for item in list(buttons):
            if key is not None and getattr(item, "key", None) == key:
                return item
            if label is not None and getattr(getattr(item, "proto", None), "label", None) == label:
                return item
        self.fail(f"Button not found: label={label!r} key={key!r}")

    def _validation_dataframe(self, app: AppTest):
        for item in list(app.dataframe):
            value = getattr(item, "value", None)
            if getattr(value, "columns", None) is not None and {"Severity", "Message"}.issubset(set(value.columns)):
                return value
        self.fail("Validation dataframe not found")

    def _select_request_domain(self, app: AppTest, domain: str) -> None:
        app.radio(key="v22_request_inputs_active_domain").set_value(domain)
        self._run(app)

    def _baseline_row_by_id(self, vde_id: int) -> dict:
        if int(vde_id) == 5001:
            row = deepcopy(_baseline_row())
            row.update(
                {
                    "id": 5001,
                    "make": "VOLVO",
                    "model": "XC40",
                    "year": 2028,
                    "mass_kg": 1700.0,
                    "test_mass_kg": 1800.0,
                    "inertia_class": 1814.0,
                    "vde_total_mj_per_km": 1.345,
                    "vde_net_mj_per_km": 1.222,
                }
            )
            return row
        row = deepcopy(_baseline_row())
        row["vde_total_mj_per_km"] = 1.234
        row["vde_net_mj_per_km"] = 1.111
        return row

    def _component_lookup_sample_rows(self, domain: str) -> list[dict]:
        if domain == "brake":
            return [
                {
                    "lookup_id": "BRAKE-MOCK-LOW",
                    "ID": "BRAKE-MOCK-LOW",
                    "Code / Name": "Brake Low",
                    "A": 1.5,
                    "B": 0.0005,
                    "C": 0.0001,
                    "Status": "active",
                    "Source": "qa",
                    "_raw": {
                        "component_id": "BRAKE-MOCK-LOW",
                        "component_name": "Brake Low",
                        "brake_A": 1.5,
                        "brake_B": 0.0005,
                        "brake_C": 0.0001,
                    },
                },
                {
                    "lookup_id": "BRAKE-MOCK-HIGH",
                    "ID": "BRAKE-MOCK-HIGH",
                    "Code / Name": "Brake High",
                    "A": 4.5,
                    "B": 0.0015,
                    "C": 0.0003,
                    "Status": "active",
                    "Source": "qa",
                    "_raw": {
                        "component_id": "BRAKE-MOCK-HIGH",
                        "component_name": "Brake High",
                        "brake_A": 4.5,
                        "brake_B": 0.0015,
                        "brake_C": 0.0003,
                    },
                },
            ]
        if domain == "transmission":
            return [
                {
                    "lookup_id": "TRANS-MOCK-LOW",
                    "ID": "TRANS-MOCK-LOW",
                    "Code / Name": "Transmission Low",
                    "A": 1.2,
                    "B": 0.0002,
                    "C": 0.00005,
                    "Status": "active",
                    "Source": "qa",
                    "_raw": {
                        "component_id": "TRANS-MOCK-LOW",
                        "component_name": "Transmission Low",
                        "trans_A": 1.2,
                        "trans_B": 0.0002,
                        "trans_C": 0.00005,
                    },
                },
                {
                    "lookup_id": "TRANS-MOCK-HIGH",
                    "ID": "TRANS-MOCK-HIGH",
                    "Code / Name": "Transmission High",
                    "A": 3.8,
                    "B": 0.0009,
                    "C": 0.0002,
                    "Status": "active",
                    "Source": "qa",
                    "_raw": {
                        "component_id": "TRANS-MOCK-HIGH",
                        "component_name": "Transmission High",
                        "trans_A": 3.8,
                        "trans_B": 0.0009,
                        "trans_C": 0.0002,
                    },
                },
            ]
        raise ValueError(f"Unsupported sample lookup domain: {domain}")

    def test_active_section_router_only_calls_selected_renderer(self):
        calls = []

        def renderer(name):
            def _inner():
                calls.append(name)
                return name

            return _inner

        result = render_active_v22_section(
            "inputs",
            {
                "baseline": renderer("baseline"),
                "matrix": renderer("matrix"),
                "inputs": renderer("inputs"),
                "preview": renderer("preview"),
            },
        )
        self.assertEqual(result, "inputs")
        self.assertEqual(calls, ["inputs"])

    def test_v22_component_static_guard_no_forbidden_integrations(self):
        text = COMPONENT_PATH.read_text(encoding="utf-8")
        forbidden = [
            "save_vde_setup_result",
            "execute_vde_request_save_plan",
            "generate_vde_request_report_xlsx",
            "render_vde_setup_workbook_v21",
            "vde_setup_workbook_v21",
        ]
        for token in forbidden:
            self.assertNotIn(token, text)

    def test_v22_new_modules_do_not_import_vde_setup(self):
        for path in (COMPONENT_PATH, STYLE_PATH, VIEWMODEL_PATH):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("src.vde_app.components.vde_setup", text)

    def test_v22_page_static_guard_only_uses_v22_session_root(self):
        text = PAGE_PATH.read_text(encoding="utf-8")
        self.assertIn('page_title="EcoDrive - VDE Setup"', text)
        self.assertIn('st.header("VDE Setup")', text)
        self.assertIn("Baseline, request, scenario, and engineering review workflow.", text)
        self.assertNotIn("vde_setup_workbook_v21", text)
        self.assertNotIn("render_vde_setup_workbook_v21", text)

    def test_v21_and_v22_pages_compile(self):
        for path in (V21_PAGE_PATH, PAGE_PATH, COMPONENT_PATH, STYLE_PATH, VIEWMODEL_PATH):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_page_without_baseline_renders_only_baseline_section(self):
        app = self._app()
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

        self.assertFalse(app.exception)
        self.assertIn("Baseline", [item.value for item in app.subheader])
        self.assertNotIn("Proposal Matrix", [item.value for item in app.subheader])
        self.assertTrue(any("No baseline loaded." in getattr(item, "value", "") for item in app.info))

    def test_baseline_new_test_fresh_form_renders_without_crash(self):
        app = self._app()
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=[]):
            self._run(app)
            app.radio(key="v22_baseline_source_selector").set_value("New Test")
            self._run(app)

        self.assertFalse(app.exception)
        self.assertIsNone(app.number_input(key="v22_baseline_new_test__A").value)
        self.assertIsNone(app.number_input(key="v22_baseline_new_test__B").value)
        self.assertIsNone(app.number_input(key="v22_baseline_new_test__C").value)
        self.assertIsNone(app.number_input(key="v22_baseline_new_test__test_mass_kg").value)

    def test_baseline_new_test_numeric_state_is_rehydrated_and_zero_is_preserved(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = apply_v22_new_test_baseline(
            create_v22_state(),
            {
                "A": 120.0,
                "B": 0.0,
                "C": 0.008,
                "test_mass_kg": 1600.0,
                "legislation": "EPA",
                "cycle_name": "FTP75_HWFET",
            },
        )
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=[]):
            self._run(app)
            self._run(app)

        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key="v22_baseline_new_test__A").value, 120.0)
        self.assertEqual(app.number_input(key="v22_baseline_new_test__B").value, 0.0)
        self.assertEqual(app.number_input(key="v22_baseline_new_test__C").value, 0.008)
        self.assertEqual(app.number_input(key="v22_baseline_new_test__test_mass_kg").value, 1600.0)

    def test_baseline_new_test_legacy_tuple_is_sanitized_before_number_input(self):
        app = self._app()
        state = create_v22_state()
        state["baseline"]["source_type"] = "NEW_TEST"
        app.session_state[V22_SESSION_KEY] = state
        app.session_state["v22_baseline_new_test__A"] = ()
        app.session_state["v22_baseline_new_test__B"] = ()
        app.session_state["v22_baseline_new_test__C"] = ()
        app.session_state["v22_baseline_new_test__test_mass_kg"] = ()
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=[]):
            self._run(app)

        self.assertFalse(app.exception)
        self.assertIsNone(app.number_input(key="v22_baseline_new_test__A").value)
        self.assertIsNone(app.number_input(key="v22_baseline_new_test__B").value)
        self.assertIsNone(app.number_input(key="v22_baseline_new_test__C").value)
        self.assertIsNone(app.number_input(key="v22_baseline_new_test__test_mass_kg").value)

    def test_page_with_loaded_baseline_shows_branding_and_sidebar_flow(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = self._loaded_state()
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

        self.assertFalse(app.exception)
        sidebar_labels = [item.label for item in app.sidebar.button if getattr(item, "label", None)]
        self.assertIn("Reset VDE Setup request", sidebar_labels)
        self.assertTrue(any("Baseline & Corrections" in label for label in sidebar_labels))
        self.assertTrue(any("Request Inputs" in label for label in sidebar_labels))
        markdown_values = [item.value for item in app.markdown]
        self.assertTrue(any("AUDI" in value for value in markdown_values))
        self.assertTrue(any("EPA" in value for value in markdown_values))

    def test_page_navigation_previous_next_and_reset_keep_only_active_section(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = self._loaded_state()
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

            self._button(app.sidebar.button, key="v22_sidebar_nav__matrix").click()
            self._run(app)
        self.assertIn("Proposal Matrix", [item.value for item in app.subheader])
        self.assertNotIn("Baseline", [item.value for item in app.subheader])

        self._button(app.button, label="Next").click()
        self._run(app)
        self.assertIn("Request Inputs", [item.value for item in app.subheader])
        self.assertNotIn("Proposal Matrix", [item.value for item in app.subheader])

        self._button(app.sidebar.button, key="v22_sidebar_nav__preview").click()
        self._run(app)
        self.assertIn("Preview & Save", [item.value for item in app.subheader])
        self.assertNotIn("Request Inputs", [item.value for item in app.subheader])

        self._button(app.button, label="Previous").click()
        self._run(app)
        self.assertIn("Request Inputs", [item.value for item in app.subheader])

        self._button(app.sidebar.button, key="reset_vde_setup_v22").click()
        self._run(app)
        self.assertFalse(app.session_state[V22_SESSION_KEY]["baseline"]["loaded"])
        self.assertEqual(app.session_state[V22_SESSION_KEY]["active_section"], "baseline")
        self.assertIn("Baseline", [item.value for item in app.subheader])

    def test_page_matrix_inputs_and_preview_actions_still_work(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = self._loaded_state()
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

            self._button(app.sidebar.button, key="v22_sidebar_nav__matrix").click()
            self._run(app)
        app.selectbox(key="v22_matrix_requested_1_mass").select("Custom test mass")
        self._button(app.button, label="Apply Proposal Matrix").click()
        self._run(app)
        self.assertEqual(
            app.session_state[V22_SESSION_KEY]["proposals"][0]["domains"]["mass"]["selection_mode"],
            "Custom test mass",
        )

        self._button(app.button, label="Next").click()
        self._run(app)
        app.session_state[V22_SESSION_KEY] = apply_v22_domain_inputs(
            app.session_state[V22_SESSION_KEY],
            "mass",
            {"requested_1": {"test_mass_kg": 1810.0}},
        )
        self._run(app)
        self.assertEqual(
            app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"]["mass"]["test_mass_kg"],
            1810.0,
        )

        self._button(app.button, label="Next: Preview").click()
        self._run(app)
        self._button(app.button, key="v22_validate_preview").click()
        self._run(app)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["preview"]["status"], "fresh")
        self.assertTrue(app.session_state[V22_SESSION_KEY]["preview"]["result"])

    def test_request_input_pager_moves_between_direct_domains_without_applying(self):
        app = self._app()
        state = apply_v22_proposal_matrix(
            self._loaded_state(),
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Custom test mass",
                    "aero": "Delta CdA",
                }
            ],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

        preview_before = deepcopy(app.session_state[V22_SESSION_KEY]["preview"])
        self._button(app.button, label="Next: Aero").click()
        self._run(app)
        self.assertEqual(app.radio(key="v22_request_inputs_active_domain").value, "aero")
        self.assertEqual(app.session_state[V22_SESSION_KEY]["preview"], preview_before)

        self._button(app.button, label="Next: Preview").click()
        self._run(app)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["active_section"], "preview")
        self.assertEqual(app.session_state[V22_SESSION_KEY]["preview"], preview_before)

    def test_page_defaults_unit_system_to_metric(self):
        app = self._app()

        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

        self.assertEqual(app.session_state["unit_system"], "Metric")

    def test_page_respects_preselected_us_customary_units(self):
        app = self._app()
        app.session_state["unit_system"] = "US customary"

        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

        self.assertEqual(app.session_state["unit_system"], "US customary")

    def test_baseline_candidate_change_does_not_mutate_loaded_state_or_branding_until_load(self):
        app = self._app()
        state = self._loaded_state()
        state["preview"] = {"status": "fresh", "fingerprint": "fp-baseline", "result": {"validation_summary": {}}}
        app.session_state[V22_SESSION_KEY] = state
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)
            before = deepcopy(app.session_state[V22_SESSION_KEY])

            app.selectbox(key="v22_baseline_selector").select(5001)
            self._run(app)

        after = app.session_state[V22_SESSION_KEY]
        self.assertEqual(after, before)
        self.assertEqual(after["baseline"]["selected_vde_id"], 4998)
        self.assertEqual(after["preview"]["fingerprint"], "fp-baseline")
        markdown_values = [getattr(item, "value", "") for item in app.markdown]
        self.assertTrue(any("Selected candidate differs from the loaded baseline." in value for value in markdown_values))
        self.assertTrue(any("AUDI" in value for value in markdown_values))
        self.assertFalse(any("VOLVO" in value and "Loaded Baseline Context" in value for value in markdown_values))

    def test_baseline_load_button_updates_loaded_baseline_and_branding(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = self._loaded_state()
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()), patch.object(vde_request_compact, "fetch_vde_by_id", side_effect=self._baseline_row_by_id):
            self._run(app)
            app.selectbox(key="v22_baseline_selector").select(5001)
            self._run(app)
            self._button(app.button, key="v22_load_baseline").click()
            self._run(app)

        self.assertEqual(app.session_state[V22_SESSION_KEY]["baseline"]["selected_vde_id"], 5001)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["baseline"]["printed"]["make"], "VOLVO")
        markdown_values = [getattr(item, "value", "") for item in app.markdown]
        self.assertTrue(any("VOLVO" in value for value in markdown_values))

    def test_baseline_browser_empty_does_not_remove_loaded_summary(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = self._loaded_state()
        app.session_state["v22_filter_model"] = "ZZZ"
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

        self.assertEqual(app.session_state[V22_SESSION_KEY]["baseline"]["selected_vde_id"], 4998)
        markdown_values = [getattr(item, "value", "") for item in app.markdown]
        self.assertTrue(any("Loaded Baseline" in value for value in markdown_values))
        self.assertTrue(any("AUDI" in value for value in markdown_values))
        self.assertTrue(any("No baseline rows match the current filters." in getattr(item, "value", "") for item in app.info))

    def test_baseline_browser_switches_runtime_db_without_cache_or_filter_leakage(self):
        db_a = self._temp_db_path()
        db_b = self._temp_db_path()
        self._seed_baseline_browser_db(db_a, vde_id=910001, make="QA-A", model="BASELINE-A-ONLY")
        self._seed_baseline_browser_db(db_b, vde_id=910002, make="QA-B", model="BASELINE-B-ONLY")

        vde_request_compact._baseline_summary_rows.clear()

        app = self._app()
        app.session_state["ctx"] = {"db_path": str(db_a)}
        self._run(app)

        self.assertFalse(app.exception)
        browser_df = next(
            item.value
            for item in app.dataframe
            if getattr(getattr(item, "value", None), "columns", None) is not None
            and {"VDE ID", "Make", "Model", "Legislation"}.issubset(set(item.value.columns))
        )
        self.assertEqual(set(browser_df["VDE ID"]), {910001})
        self.assertEqual(set(browser_df["Model"]), {"BASELINE-A-ONLY"})
        self.assertTrue(any(str(db_a.resolve()) in str(getattr(item, "value", "")) for item in app.caption))

        app.text_input(key="v22_filter_model").set_value("BASELINE-A-ONLY")
        self._run(app)
        filtered_df = next(
            item.value
            for item in app.dataframe
            if getattr(getattr(item, "value", None), "columns", None) is not None
            and {"VDE ID", "Make", "Model", "Legislation"}.issubset(set(item.value.columns))
        )
        self.assertEqual(set(filtered_df["Model"]), {"BASELINE-A-ONLY"})

        app.text_input(key="v22_runtime_db_path").set_value(str(db_b))
        self._run(app)

        switched_df = next(
            item.value
            for item in app.dataframe
            if getattr(getattr(item, "value", None), "columns", None) is not None
            and {"VDE ID", "Make", "Model", "Legislation"}.issubset(set(item.value.columns))
        )
        self.assertEqual(set(switched_df["VDE ID"]), {910002})
        self.assertEqual(set(switched_df["Model"]), {"BASELINE-B-ONLY"})
        self.assertEqual(app.text_input(key="v22_filter_model").value, "")
        self.assertEqual(Path(app.session_state["_active_runtime_db_path"]).resolve(), db_b.resolve())
        self.assertTrue(any(str(db_b.resolve()) in str(getattr(item, "value", "")) for item in app.caption))

    def test_baseline_loaded_summary_and_corrections_follow_units_and_remain_read_only(self):
        app = self._app()
        state = self._loaded_state()
        state["baseline"]["effective"]["test_mass_basis"] = "EPA_INERTIA_CLASS"
        state["baseline"]["effective"]["vde_total_mj_per_km"] = 1.234
        state["baseline"]["effective"]["vde_net_mj_per_km"] = 1.111
        state["baseline"]["corrections"] = {"inertia_class": 1928.0, "cda_m2": 0.61}
        state["baseline"]["effective"]["inertia_class"] = 1928.0
        state["baseline"]["effective"]["cda_m2"] = 0.61
        app.session_state[V22_SESSION_KEY] = state
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

        dataframes = [item.value for item in list(app.dataframe)]
        correction_df = next(df for df in dataframes if getattr(df, "columns", None) is not None and "Printed" in df.columns and "Effective" in df.columns and "Domain" in df.columns)
        self.assertEqual(correction_df.iloc[0]["Field"], "EPA ETW / TWC")
        self.assertEqual(correction_df.iloc[0]["Effective"], "1928")
        self.assertFalse(hasattr(app, "data_editor") and len(app.data_editor) > 0)

        app.selectbox(key="unit_system").select("US customary")
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

        us_dataframes = [item.value for item in list(app.dataframe)]
        us_correction_df = next(df for df in us_dataframes if getattr(df, "columns", None) is not None and "Printed" in df.columns and "Effective" in df.columns and "Domain" in df.columns)
        self.assertEqual(us_correction_df.iloc[0]["Effective"], "4251")

    def test_request_inputs_caption_and_mass_widget_follow_selected_units(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = self._loaded_state()
        self._run(app)

        self._button(app.sidebar.button, key="v22_sidebar_nav__matrix").click()
        self._run(app)
        app.selectbox(key="v22_matrix_requested_1_mass").select("Performance loaded mass")
        self._button(app.button, label="Apply Proposal Matrix").click()
        self._run(app)

        self._button(app.sidebar.button, key="v22_sidebar_nav__inputs").click()
        self._run(app)

        captions = [getattr(item, "value", "") for item in list(getattr(app, "caption", []))]
        self.assertTrue(any("Configure and apply one engineering domain at a time." in value for value in captions))
        self.assertFalse(any("Applied state remains canonical." in value for value in captions))
        self.assertIn("kg", captions)
        self.assertEqual(app.number_input(key="v22_simple_mass__requested_1__mass_kg").step, 1.0)

        app.selectbox(key="unit_system").select("US customary")
        self._run(app)

        us_captions = [getattr(item, "value", "") for item in list(getattr(app, "caption", []))]
        self.assertIn("lb", us_captions)
        self.assertEqual(app.number_input(key="v22_simple_mass__requested_1__mass_kg").step, 1.0)

    def test_request_inputs_pressure_widget_step_matches_visual_units(self):
        app = self._app()
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        app.session_state["v22_inputs_domain"] = "Tire"
        self._run(app)

        self.assertEqual(app.radio(key="v22_tire_pressure_unit").value, "kPa")
        self.assertEqual(app.number_input(key="v22_simple_tire__requested_1__front_pressure_psi").step, 1.0)

        app.radio(key="v22_tire_pressure_unit").set_value("psi")
        self._run(app)

        self.assertEqual(app.number_input(key="v22_simple_tire__requested_1__front_pressure_psi").step, 0.5)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["ui_preferences"]["tire_pressure_unit"], "psi")

        app.selectbox(key="unit_system").select("US customary")
        self._run(app)

        self.assertEqual(app.radio(key="v22_tire_pressure_unit").value, "psi")
        self.assertEqual(app.number_input(key="v22_simple_tire__requested_1__front_pressure_psi").step, 0.5)

    def test_tire_pressure_unit_change_preserves_unapplied_edit_and_runtime_state(self):
        app = self._app()
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}],
        )
        state = apply_v22_domain_inputs(
            state,
            "tire",
            {"requested_1": {"front_pressure_psi": 39.0, "rear_pressure_psi": 39.0}},
        )
        state["preview"] = {"status": "fresh", "fingerprint": "fp-pressure", "result": {"validation_summary": {}}}
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        app.session_state["v22_inputs_domain"] = "Tire"
        self._run(app)

        revision_before = app.session_state[V22_SESSION_KEY]["domain_input_state"]["tire"]["revision"]
        applied_before = app.session_state[V22_SESSION_KEY]["domain_input_state"]["tire"]["last_applied_at"]
        preview_before = deepcopy(app.session_state[V22_SESSION_KEY]["preview"])

        app.session_state["v22_simple_tire__requested_1__front_pressure_psi"] = 269.0
        app.session_state["v22_simple_tire__requested_1__rear_pressure_psi"] = 269.0
        self._run(app)

        app.radio(key="v22_tire_pressure_unit").set_value("psi")
        self._run(app)

        self.assertAlmostEqual(app.session_state["v22_simple_tire__requested_1__front_pressure_psi"], 39.0151514494, places=3)
        self.assertAlmostEqual(app.session_state["v22_simple_tire__requested_1__rear_pressure_psi"], 39.0151514494, places=3)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["domain_input_state"]["tire"]["revision"], revision_before)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["domain_input_state"]["tire"]["last_applied_at"], applied_before)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["preview"], preview_before)

    def test_request_inputs_overview_and_domain_header_show_active_domain_context(self):
        app = self._app()
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        self._run(app)

        markdown_values = [getattr(item, "value", "") for item in app.markdown]
        self.assertTrue(any("1 direct domains | 1 applied" in value for value in markdown_values))
        self.assertTrue(any("Applied" in value for value in markdown_values))

    def test_request_inputs_only_renders_apply_button_for_active_domains(self):
        app = self._app()
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Custom test mass",
                    "aero": "Inherit",
                    "brake": "Not used",
                }
            ],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        self._run(app)

        apply_buttons = [getattr(getattr(item, "proto", None), "label", "") for item in list(app.button) if getattr(getattr(item, "proto", None), "label", "").startswith("Apply ")]
        self.assertEqual(apply_buttons.count("Apply Mass"), 1)
        self.assertNotIn("Apply Aero", apply_buttons)
        self.assertNotIn("Apply Brake", apply_buttons)

    def test_request_inputs_domain_selector_renders_only_selected_editor(self):
        app = self._app()
        state = apply_v22_proposal_matrix(
            self._loaded_state(),
            [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass", "brake": "Absolute ABC"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        self._run(app)

        self.assertIsNotNone(app.number_input(key="v22_simple_mass__requested_1__test_mass_kg"))
        self.assertNotIn("v22_simple_brake__requested_1__brake_A_coef_N", [item.key for item in app.number_input])
        self._select_request_domain(app, "brake")
        self.assertIsNotNone(app.number_input(key="v22_simple_brake__requested_1__brake_A_coef_N"))
        self.assertNotIn("v22_simple_mass__requested_1__test_mass_kg", [item.key for item in app.number_input])

    def test_gvwr_mass_sheet_exposes_curb_and_payload_inputs(self):
        app = self._app()
        state = apply_v22_proposal_matrix(
            self._loaded_state(),
            [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "GVWR loaded mass"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        self._run(app)

        self.assertIsNotNone(app.number_input(key="v22_simple_mass__requested_1__mass_kg"))
        self.assertIsNotNone(app.number_input(key="v22_simple_mass__requested_1__payload_kg"))
        self.assertNotIn("v22_simple_mass__requested_1__gvwr_kg", [item.key for item in app.number_input])

    def test_request_inputs_mass_weight_distribution_renders_and_applies(self):
        app = self._app()
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state

        self._run(app)

        self.assertEqual(app.number_input(key="v22_simple_mass__requested_1__weight_dist_fr_pct").step, 0.1)
        self.assertEqual(app.number_input(key="v22_correction__mass__weight_dist_fr_pct").step, 0.1)

        app.number_input(key="v22_simple_mass__requested_1__mass_kg").set_value(1340)
        app.number_input(key="v22_simple_mass__requested_1__weight_dist_fr_pct").set_value(60.0)
        self._button(app.button, label="Apply Mass").click()
        self._run(app)

        applied_inputs = app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"]["mass"]
        self.assertEqual(applied_inputs["weight_dist_fr_pct"], 60.0)
        self.assertEqual(applied_inputs["mass_kg"], 1340.0)

    def test_preview_structured_validation_messages_follow_unit_toggle_with_safe_text_fallback(self):
        app = self._app()
        state = self._loaded_state()
        state["active_section"] = "preview"
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "fp-issue-toggle",
            "result": {
                "fingerprint": "fp-issue-toggle",
                "validation_summary": {
                    "overall_status": "Review",
                    "ok_count": 0,
                    "review_count": 1,
                    "missing_count": 0,
                    "invalid_count": 1,
                    "blocked_count": 0,
                    "warning_count": 0,
                },
                "comparison_rows": [],
                "proposal_models": [],
                "resolution_result": {
                    "issues": [
                        {
                            "severity": "INVALID",
                            "field_key": "target_curb_mass_kg",
                            "actual": 1814.0,
                            "min": 1700.0,
                            "max": 1800.0,
                            "status": "INVALID",
                        },
                        {
                            "severity": "REVIEW",
                            "field_key": "mass_rule_notes",
                            "message": "Fallback text stays untouched.",
                        },
                    ],
                    "proposal_results": [],
                },
            },
        }
        issues = state["preview"]["result"]["resolution_result"]["issues"]
        self.assertEqual(vde_request_compact.format_v22_issue_for_display(issues[0], "Metric"), "Curb mass 1814 kg is outside the allowed interval (1700, 1800] kg.")
        self.assertEqual(vde_request_compact.format_v22_issue_for_display(issues[1], "Metric"), "Fallback text stays untouched.")
        self.assertEqual(vde_request_compact.format_v22_issue_for_display(issues[0], "US customary"), "Curb mass 3999 lb is outside the allowed interval (3748, 3968] lb.")

    def test_preview_empty_state_shows_validate_action_and_save_gate(self):
        app = self._app()
        state = self._loaded_state()
        state["active_section"] = "preview"
        app.session_state[V22_SESSION_KEY] = state
        self._run(app)

        info_messages = [getattr(item, "value", "") for item in list(getattr(app, "info", []))]
        self.assertTrue(any("No preview generated yet." in value for value in info_messages))
        self._button(app.button, key="v22_validate_preview")
        self.assertFalse(any(getattr(item, "key", None) == "v22_save_request" for item in app.button))

    def test_fresh_preview_reorganized_sections_show_baseline_requested_and_save_gate(self):
        app = self._app()
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass", "aero": "Delta CdA"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit", "aero": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})
        state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"delta_CdA": -0.01}})
        state["active_section"] = "preview"
        app.session_state[V22_SESSION_KEY] = state
        self._run(app)
        self._button(app.button, key="v22_validate_preview").click()
        self._run(app)

        markdown_values = [getattr(item, "value", "") for item in app.markdown]
        self.assertTrue(any("Scenario Overview" in value for value in markdown_values))
        self.assertTrue(any("Requested #2" in value for value in markdown_values))

        dataframes = [item.value for item in list(app.dataframe)]
        comparison_df = next(df for df in dataframes if getattr(df, "columns", None) is not None and "Effective Baseline" in df.columns and "Requested #1" in df.columns)
        self.assertIn("Requested #2", set(comparison_df.columns))
        self.assertEqual(app.selectbox(key="v22_roadload_max_speed").value, 140)
        # AppTest materializes widgets in inactive tabs; the save control remains
        # visually contained by the DB Preview & Save tab.
        self.assertTrue(any(getattr(item, "key", None) == "v22_save_request" for item in app.button))

    def test_stale_preview_warning_is_visible_without_regeneration(self):
        app = self._app()
        state = self._loaded_state()
        state["active_section"] = "preview"
        state["preview"] = {
            "status": "stale",
            "fingerprint": "fp-stale",
            "result": {
                "fingerprint": "fp-stale",
                "validation_summary": {"overall_status": "Review", "proposal_count": 0, "ok_count": 0, "review_count": 0, "missing_count": 0, "invalid_count": 0, "blocked_count": 0, "warning_count": 0},
                "comparison_rows": [],
                "proposal_models": [],
                "resolution_result": {"issues": [], "proposal_results": []},
            },
        }
        app.session_state[V22_SESSION_KEY] = state
        self._run(app)

        markdown_values = [getattr(item, "value", "") for item in app.markdown]
        self.assertTrue(any("Preview is stale." in value for value in markdown_values))
        self.assertEqual(app.session_state[V22_SESSION_KEY]["preview"]["fingerprint"], "fp-stale")

    def test_switching_display_units_preserves_v22_state_and_preview_fingerprint(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = self._loaded_state()
        self._run(app)

        self._button(app.sidebar.button, key="v22_sidebar_nav__matrix").click()
        self._run(app)
        app.selectbox(key="v22_matrix_requested_1_mass").select("Custom test mass")
        self._button(app.button, label="Apply Proposal Matrix").click()
        self._run(app)

        self._button(app.button, label="Next").click()
        self._run(app)
        app.session_state[V22_SESSION_KEY] = apply_v22_domain_inputs(
            app.session_state[V22_SESSION_KEY],
            "mass",
            {"requested_1": {"test_mass_kg": 1810.0}},
        )
        self._run(app)

        self._button(app.button, label="Next: Preview").click()
        self._run(app)
        self._button(app.button, key="v22_validate_preview").click()
        self._run(app)

        state_before = deepcopy(app.session_state[V22_SESSION_KEY])
        fingerprint_before = app.session_state[V22_SESSION_KEY]["preview"]["fingerprint"]

        app.selectbox(key="unit_system").select("US customary")
        self._run(app)

        self.assertEqual(app.session_state["unit_system"], "US customary")
        self.assertEqual(app.session_state[V22_SESSION_KEY], state_before)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["preview"]["fingerprint"], fingerprint_before)
        self.assertEqual(app.session_state[V22_SESSION_KEY]["preview"]["status"], "fresh")
        self.assertEqual(app.session_state[V22_SESSION_KEY]["active_section"], "preview")

    def test_switching_units_discards_pending_input_and_rebuilds_widget_from_canonical_state(self):
        app = self._app()
        app.session_state[V22_SESSION_KEY] = self._loaded_state()
        self._run(app)

        self._button(app.sidebar.button, key="v22_sidebar_nav__matrix").click()
        self._run(app)
        app.selectbox(key="v22_matrix_requested_1_mass").select("Performance loaded mass")
        self._button(app.button, label="Apply Proposal Matrix").click()
        self._run(app)

        self._button(app.button, label="Next").click()
        self._run(app)
        app.session_state[V22_SESSION_KEY] = apply_v22_domain_inputs(
            app.session_state[V22_SESSION_KEY],
            "mass",
            {"requested_1": {"mass_kg": 1810.0}},
        )
        app.session_state["v22_simple_mass__requested_1__mass_kg"] = 1900.0
        self._run(app)

        app.selectbox(key="unit_system").select("US customary")
        self._run(app)

        self.assertAlmostEqual(app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"]["mass"]["mass_kg"], 1810.0, places=6)
        self.assertAlmostEqual(app.session_state["v22_simple_mass__requested_1__mass_kg"], 3990.3679455485, places=2)
        info_messages = [getattr(item, "value", "") for item in list(getattr(app, "info", []))]
        self.assertTrue(any("Unapplied edits were reset when display units changed." in value for value in info_messages))

    def test_mass_form_submit_persists_target_curb_mass_and_walk_from_shift_uses_resolved_class(self):
        app = self._app()
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1480.0,
                "test_mass_kg": 1480.0,
                "inertia_class": 1588.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "TWC shift / target class"},
            ],
        )
        app.session_state[V22_SESSION_KEY] = state
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)
            self._button(app.sidebar.button, key="v22_sidebar_nav__inputs").click()
            self._run(app)

            app.number_input(key="v22_simple_mass__requested_1__mass_kg").set_value(1200).run()
            app.selectbox(key="v22_simple_mass__requested_2__shift_steps").select("+1")
            self._button(app.button, label="Apply Mass").click()
            self._run(app)

            stored_state = app.session_state[V22_SESSION_KEY]
            self.assertEqual(stored_state["proposals"][0]["inputs"]["mass"]["mass_kg"], 1200.0)
            self.assertEqual(stored_state["proposals"][1]["inputs"]["mass"]["shift_steps"], 1.0)
            self.assertEqual(stored_state["proposals"][1]["inputs"]["mass"]["curb_position"], "Top")

            self._button(app.button, label="Next: Preview").click()
            self._run(app)
            self._button(app.button, key="v22_validate_preview").click()
            self._run(app)

            bundle = app.session_state[V22_SESSION_KEY]["preview"]["result"]
            proposal_results = list(bundle["resolution_result"]["proposal_results"] or [])
            req1_snapshot = dict(proposal_results[0]["resolved_snapshot"] or {})
            req2_snapshot = dict(proposal_results[1]["resolved_snapshot"] or {})

            self.assertEqual(req1_snapshot["mass_kg"], 1200.0)
            self.assertEqual(req1_snapshot["inertia_class"], 1361.0)
            self.assertEqual(req2_snapshot["inertia_class"], 1417.0)
            self.assertEqual(req2_snapshot["test_mass_kg"], 1446.0)

    def test_request_inputs_apply_real_form_widgets_across_domains(self):
        app = self._app()
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1500.0,
                "test_mass_kg": 1500.0,
                "inertia_class": 1588.0,
                "cda_m2": 0.62,
                "rrc_N_per_kN": 8.1,
                "tire_A_final": 40.0,
                "tire_B_final": 0.001,
                "tire_C_final": 0.0001,
                "trans_A_coef_N": 6.0,
                "trans_B_coef_Npkph": 0.003,
                "trans_C_coef_Npkph2": 0.001,
                "brake_A_coef_N": 4.0,
                "brake_B_Npkph": 0.0008,
                "brake_C_coef_Npkph2": 0.0001,
                "axle_hub_A": 3.0,
                "axle_hub_B": 0.0004,
                "axle_hub_C": 0.0001,
                "parasitic_A_coef_N": 5.0,
                "parasitic_B_Npkph": 0.0007,
                "parasitic_C_coef_Npkph2": 0.0002,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Curb mass -> EPA TWC",
                    "aero": "Absolute CdA",
                    "tire": "Target final RRC",
                    "transmission": "Absolute ABC",
                    "brake": "Absolute ABC",
                    "axle_hubs": "Absolute ABC",
                    "parasitic": "Delta ABC",
                }
            ],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state

        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

            app.number_input(key="v22_simple_mass__requested_1__mass_kg").set_value(1340)
            self._button(app.button, label="Apply Mass").click()
            self._run(app)
            mass_state = app.session_state[V22_SESSION_KEY]
            self.assertEqual(mass_state["proposals"][0]["inputs"]["mass"]["mass_kg"], 1340.0)
            mass_status = mass_state["domain_input_state"]["mass"]["proposal_statuses"]["requested_1"]
            self.assertNotEqual(mass_status["status"], "missing")
            self.assertNotIn("Curb mass is required", str(mass_status))

            self._select_request_domain(app, "aero")
            app.number_input(key="v22_simple_aero__requested_1__cda_m2").set_value(0.67)
            self._button(app.button, label="Apply Aero").click()
            self._run(app)
            aero_state = app.session_state[V22_SESSION_KEY]
            self.assertEqual(aero_state["proposals"][0]["inputs"]["aero"]["cda_m2"], 0.67)
            self.assertNotIn("New CdA is required", str(aero_state["domain_input_state"]["aero"]["proposal_statuses"]["requested_1"]))

            self._select_request_domain(app, "tire")
            app.number_input(key="v22_simple_tire__requested_1__target_rrc_N_per_kN").set_value(7.0)
            self._button(app.button, label="Apply Tire").click()
            self._run(app)
            self.assertEqual(app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"]["tire"]["target_rrc_N_per_kN"], 7.0)

            self._select_request_domain(app, "transmission")
            app.number_input(key="v22_simple_transmission__requested_1__trans_A_coef_N").set_value(4.0)
            app.number_input(key="v22_simple_transmission__requested_1__trans_B_coef_Npkph").set_value(0.0)
            app.number_input(key="v22_simple_transmission__requested_1__trans_C_coef_Npkph2").set_value(0.0)
            self._button(app.button, label="Apply Transmission").click()
            self._run(app)
            trans_inputs = app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"]["transmission"]
            self.assertEqual(
                trans_inputs,
                {
                    "trans_A_coef_N": 4.0,
                    "trans_B_coef_Npkph": 0.0,
                    "trans_C_coef_Npkph2": 0.0,
                    "transmission_application_mode": "APPLY_DELTA_TO_TOTAL",
                },
            )
            self.assertNotIn("A is required", str(app.session_state[V22_SESSION_KEY]["domain_input_state"]["transmission"]["proposal_statuses"]["requested_1"]))

            self._select_request_domain(app, "brake")
            app.number_input(key="v22_simple_brake__requested_1__brake_A_coef_N").set_value(2.0)
            app.number_input(key="v22_simple_brake__requested_1__brake_B_Npkph").set_value(0.0)
            app.number_input(key="v22_simple_brake__requested_1__brake_C_coef_Npkph2").set_value(0.0)
            self._button(app.button, label="Apply Brake").click()
            self._run(app)
            brake_inputs = app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"]["brake"]
            self.assertEqual(brake_inputs, {"brake_A_coef_N": 2.0, "brake_B_Npkph": 0.0, "brake_C_coef_Npkph2": 0.0})
            self.assertNotIn("A is required", str(app.session_state[V22_SESSION_KEY]["domain_input_state"]["brake"]["proposal_statuses"]["requested_1"]))

            self._select_request_domain(app, "axle_hubs")
            app.number_input(key="v22_simple_axle_hubs__requested_1__axle_hub_A").set_value(1.5)
            app.number_input(key="v22_simple_axle_hubs__requested_1__axle_hub_B").set_value(0.0)
            app.number_input(key="v22_simple_axle_hubs__requested_1__axle_hub_C").set_value(0.0)
            self._button(app.button, label="Apply Axle & Hubs").click()
            self._run(app)
            axle_inputs = app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"]["axle_hubs"]
            self.assertEqual(axle_inputs, {"axle_hub_A": 1.5, "axle_hub_B": 0.0, "axle_hub_C": 0.0})
            self.assertEqual(app.session_state[V22_SESSION_KEY]["domain_input_state"]["axle_hubs"]["proposal_statuses"]["requested_1"]["status"], "applied_ready")

            self._select_request_domain(app, "parasitic")
            app.number_input(key="v22_simple_parasitics__requested_1__delta_A").set_value(1.5)
            app.number_input(key="v22_simple_parasitics__requested_1__delta_B").set_value(0.0)
            app.number_input(key="v22_simple_parasitics__requested_1__delta_C").set_value(0.0)
            self._button(app.button, label="Apply Parasitics").click()
            self._run(app)
            parasitic_inputs = app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"]["parasitic"]
            self.assertEqual(parasitic_inputs, {"delta_A": 1.5, "delta_B": 0.0, "delta_C": 0.0})
            self.assertEqual(app.session_state[V22_SESSION_KEY]["domain_input_state"]["parasitic"]["proposal_statuses"]["requested_1"]["status"], "applied_ready")

        final_state = app.session_state[V22_SESSION_KEY]
        bundle = build_v22_preview_bundle(final_state, baseline_context=compact_baseline_context(final_state))
        snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        self.assertEqual(snapshot["mass_kg"], 1340.0)
        self.assertEqual(snapshot["inertia_class"], 1474.0)
        self.assertEqual(snapshot["CdA"], 0.67)
        self.assertEqual(snapshot["rrc_N_per_kN"], 7.0)
        self.assertEqual(snapshot["transmission_losses"]["abc"], {"A": 4.0, "B": 0.0, "C": 0.0})
        self.assertEqual(snapshot["brake_A"], 2.0)
        self.assertEqual(snapshot["axle_hub_A"], 1.5)
        self.assertEqual(snapshot["parasitic_A"], 6.5)

    def test_request_inputs_tire_lookup_pending_widgets_apply_and_walk_from_inherit(self):
        app = self._app()
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "rrc_N_per_kN": 8.1,
                "front_pressure_psi": 38.0,
                "rear_pressure_psi": 38.0,
                "tire_code": "BASE-TIRE",
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "tire": "Inherit"},
            ],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        self._run(app)

        vde_request_compact._apply_lookup_to_widget_state(
            app.session_state,
            "tire",
            "requested_1",
            "TIRE_DB_LOOKUP",
            "Tire DB lookup",
            {
                "tire_db_id": 77,
                "tire_code": "TIRE-QA-010",
                "tire_source_vde_id": "VDE-TIRE-77",
                "rrc_N_per_kN": 8.4,
                "front_pressure_psi": 35.0,
                "rear_pressure_psi": 35.0,
                "tire_load_mass_basis": "TEST_MASS",
            },
            unit_system="Metric",
            pressure_unit="kPa",
        )

        self._button(app.button, label="Apply Tire").click()
        self._run(app)

        applied_state = app.session_state[V22_SESSION_KEY]
        tire_inputs = applied_state["proposals"][0]["inputs"]["tire"]
        self.assertEqual(tire_inputs["tire_db_id"], 77)
        self.assertEqual(tire_inputs["tire_code"], "TIRE-QA-010")
        self.assertEqual(tire_inputs["tire_source_vde_id"], "VDE-TIRE-77")
        self.assertEqual(tire_inputs["rrc_N_per_kN"], 8.4)
        self.assertEqual(tire_inputs["front_pressure_psi"], 35.0)
        self.assertEqual(tire_inputs["rear_pressure_psi"], 35.0)
        self.assertEqual(tire_inputs["tire_load_mass_basis"], "TEST_MASS")

        bundle = build_v22_preview_bundle(applied_state, baseline_context=compact_baseline_context(applied_state))
        proposal_results = list(bundle["resolution_result"]["proposal_results"] or [])
        req1_snapshot = dict(proposal_results[0]["resolved_snapshot"] or {})
        req2_snapshot = dict(proposal_results[1]["resolved_snapshot"] or {})

        self.assertEqual(req1_snapshot["tire_code"], "TIRE-QA-010")
        self.assertEqual(req1_snapshot["rrc_N_per_kN"], 8.4)
        self.assertEqual(req1_snapshot["front_pressure_psi"], 35.0)
        self.assertEqual(req2_snapshot["tire_code"], "TIRE-QA-010")
        self.assertEqual(req2_snapshot["rrc_N_per_kN"], 8.4)

    def test_request_inputs_tire_lookup_browser_uses_tire_labels_and_browse_default(self):
        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        sample_rows = [
            {
                "lookup_id": "920102",
                "Tire ID": 920102,
                "Tire code": "QA-ECO",
                "RRC": 7.0,
                "Reference pressure": 35.0,
                "Test load": 610.0,
                "Mileage": 1000.0,
                "Description": "TIRE-MOCK-ECO",
                "_raw": {"id": 920102, "tire_test_code": "QA-ECO", "rr_n_per_kn": 7.0, "test_pressure_value": 35.0, "test_load_value": 610.0},
            }
        ]
        with patch.object(vde_request_compact, "component_lookup_rows", return_value=sample_rows):
            self._run(app)

        self.assertFalse(app.exception)
        source_radio = app.radio(key="v22_lookup_source__tire")
        self.assertEqual(source_radio.options, ["Tire Database", "Existing VDE"])
        self.assertEqual(source_radio.value, "Tire Database")
        dataframes = [item.value for item in app.dataframe if getattr(item, "value", None) is not None]
        self.assertTrue(any("Tire code" in list(df.columns) for df in dataframes))

    def test_request_inputs_tire_lookup_browser_filters_and_use_selected_tire_without_apply(self):
        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        sample_rows = [
            {
                "lookup_id": "920102",
                "Tire ID": 920102,
                "Tire code": "QA-ECO",
                "RRC": 7.0,
                "Reference pressure": 35.0,
                "Test load": 610.0,
                "Mileage": 1000.0,
                "Description": "TIRE-MOCK-ECO",
                "_raw": {"id": 920102, "tire_test_code": "QA-ECO", "rr_n_per_kn": 7.0, "test_pressure_value": 35.0, "test_load_value": 610.0},
            },
            {
                "lookup_id": "920104",
                "Tire ID": 920104,
                "Tire code": "QA-LOAD",
                "RRC": 8.8,
                "Reference pressure": 30.0,
                "Test load": 650.0,
                "Mileage": 1000.0,
                "Description": "TIRE-MOCK-LOAD",
                "_raw": {"id": 920104, "tire_test_code": "QA-LOAD", "rr_n_per_kn": 8.8, "test_pressure_value": 30.0, "test_load_value": 650.0},
            },
        ]
        with patch.object(vde_request_compact, "component_lookup_rows", return_value=sample_rows):
            self._run(app)
            app.text_input(key="v22_tire_browser_code_query").set_value("QA-LOAD")
            self._run(app)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="v22_lookup_selected__tire").value, "920104")
            self._button(app.button, label="Use selected tire").click()
            self._run(app)

        self.assertEqual(app.session_state["v22_simple_tire__requested_1__tire_db_id"], "920104")
        self.assertEqual(app.session_state["v22_simple_tire__requested_1__tire_code"], "QA-LOAD")
        self.assertEqual(
            to_canonical_field_value(
                "front_pressure_psi",
                app.session_state["v22_simple_tire__requested_1__front_pressure_psi"],
                "Metric",
                pressure_unit="kPa",
            ),
            30.0,
        )
        self.assertEqual(
            to_canonical_field_value(
                "rear_pressure_psi",
                app.session_state["v22_simple_tire__requested_1__rear_pressure_psi"],
                "Metric",
                pressure_unit="kPa",
            ),
            30.0,
        )

    def test_request_inputs_brake_lookup_apply_to_routes_selected_row_without_cross_proposal_bleed(self):
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Lookup from DB"},
                {"proposal_id": "requested_2", "walk_from": "baseline", "brake": "Lookup from DB"},
            ],
        )
        sample_rows = self._component_lookup_sample_rows("brake")
        fake_streamlit = SimpleNamespace(session_state={}, info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, rerun=lambda: None)
        with patch.object(vde_request_compact, "st", fake_streamlit):
            vde_request_compact._apply_selected_lookup_row(state, "brake", "requested_1", "Component DB", sample_rows[0], "Metric")
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_component_db_id"], "BRAKE-MOCK-LOW")
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_A_coef_N"], 1.5)
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_B_Npkph"], 0.0005)
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_C_coef_Npkph2"], 0.0001)

            vde_request_compact._apply_selected_lookup_row(state, "brake", "requested_2", "Component DB", sample_rows[1], "Metric")
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_component_db_id"], "BRAKE-MOCK-LOW")
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_A_coef_N"], 1.5)
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_B_Npkph"], 0.0005)
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_C_coef_Npkph2"], 0.0001)
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_2__brake_component_db_id"], "BRAKE-MOCK-HIGH")
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_2__brake_A_coef_N"], 4.5)
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_2__brake_B_Npkph"], 0.0015)
            self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_2__brake_C_coef_Npkph2"], 0.0003)

            staged = fake_streamlit.session_state[vde_request_compact.V22_COMPONENT_LOOKUP_DRAFTS_KEY]["brake"]
            self.assertEqual(staged["requested_1"]["brake_component_db_id"], "BRAKE-MOCK-LOW")
            self.assertEqual(staged["requested_2"]["brake_component_db_id"], "BRAKE-MOCK-HIGH")

        state = apply_v22_domain_inputs(
            state,
            "brake",
            {
                "requested_1": {
                    "brake_component_db_id": fake_streamlit.session_state["v22_simple_brake__requested_1__brake_component_db_id"],
                    "brake_A_coef_N": fake_streamlit.session_state["v22_simple_brake__requested_1__brake_A_coef_N"],
                    "brake_B_Npkph": fake_streamlit.session_state["v22_simple_brake__requested_1__brake_B_Npkph"],
                    "brake_C_coef_Npkph2": fake_streamlit.session_state["v22_simple_brake__requested_1__brake_C_coef_Npkph2"],
                },
                "requested_2": {
                    "brake_component_db_id": fake_streamlit.session_state["v22_simple_brake__requested_2__brake_component_db_id"],
                    "brake_A_coef_N": fake_streamlit.session_state["v22_simple_brake__requested_2__brake_A_coef_N"],
                    "brake_B_Npkph": fake_streamlit.session_state["v22_simple_brake__requested_2__brake_B_Npkph"],
                    "brake_C_coef_Npkph2": fake_streamlit.session_state["v22_simple_brake__requested_2__brake_C_coef_Npkph2"],
                },
            },
        )
        req1 = state["proposals"][0]["inputs"]["brake"]
        req2 = state["proposals"][1]["inputs"]["brake"]
        self.assertEqual(req1["brake_component_db_id"], "BRAKE-MOCK-LOW")
        self.assertEqual(req2["brake_component_db_id"], "BRAKE-MOCK-HIGH")
        self.assertEqual(req1["brake_A_coef_N"], 1.5)
        self.assertEqual(req2["brake_A_coef_N"], 4.5)

    def test_request_inputs_brake_lookup_reverse_order_keeps_widget_targets_isolated(self):
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Lookup from DB"},
                {"proposal_id": "requested_2", "walk_from": "baseline", "brake": "Lookup from DB"},
            ],
        )
        sample_rows = self._component_lookup_sample_rows("brake")
        fake_streamlit = SimpleNamespace(session_state={}, info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, rerun=lambda: None)
        with patch.object(vde_request_compact, "st", fake_streamlit):
            vde_request_compact._apply_selected_lookup_row(state, "brake", "requested_1", "Component DB", sample_rows[1], "Metric")
            vde_request_compact._apply_selected_lookup_row(state, "brake", "requested_2", "Component DB", sample_rows[0], "Metric")

        self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_component_db_id"], "BRAKE-MOCK-HIGH")
        self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_1__brake_A_coef_N"], 4.5)
        self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_2__brake_component_db_id"], "BRAKE-MOCK-LOW")
        self.assertEqual(fake_streamlit.session_state["v22_simple_brake__requested_2__brake_A_coef_N"], 1.5)

    def test_request_inputs_transmission_lookup_apply_to_routes_selected_row_without_cross_proposal_bleed(self):
        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Lookup from DB"},
                {"proposal_id": "requested_2", "walk_from": "baseline", "transmission": "Lookup from DB"},
            ],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        sample_rows = self._component_lookup_sample_rows("transmission")

        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()), patch.object(vde_request_compact, "component_lookup_rows", return_value=sample_rows):
            self._run(app)
            app.selectbox(key="v22_lookup_target__transmission").select("requested_2")
            app.selectbox(key="v22_lookup_selected__transmission").select("TRANS-MOCK-HIGH")
            self._run(app)
            self._button(app.button, label="Use selected row").click()
            self._run(app)

        staged = app.session_state[vde_request_compact.V22_COMPONENT_LOOKUP_DRAFTS_KEY]["transmission"]
        self.assertNotIn("requested_1", staged)
        self.assertEqual(staged["requested_2"]["transmission_component_db_id"], "TRANS-MOCK-HIGH")
        self.assertEqual(staged["requested_2"]["trans_A_coef_N"], 3.8)
        self.assertNotIn("v22_simple_transmission__requested_2__trans_A_coef_N", [item.key for item in app.number_input])
        self.assertNotIn("tire", app.session_state[V22_SESSION_KEY]["proposals"][0]["inputs"])

    def test_request_inputs_tire_lookup_existing_vde_source_remains_available(self):
        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        sample_vde_rows = [
            {
                "lookup_id": 5001,
                "VDE ID": 5001,
                "Make": "VOLVO",
                "Model": "XC40",
                "Tire code": "QA-BASE",
                "RRC": 8.0,
                "Reference pressure": 38.0,
                "Description": "Existing VDE tire",
                "_raw": {"id": 5001, "tire_db_id": 920101, "tire_code": "QA-BASE", "rrc_N_per_kN": 8.0, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
            }
        ]
        with patch.object(vde_request_compact, "vde_lookup_rows", return_value=sample_vde_rows):
            self._run(app)
            app.radio(key="v22_lookup_source__tire").set_value("Existing VDE")
            self._run(app)

        self.assertFalse(app.exception)
        self.assertTrue(any("Tire code" in list(item.value.columns) for item in app.dataframe if getattr(getattr(item, "value", None), "columns", None) is not None))

    def test_request_inputs_tire_lookup_browser_shows_rows_and_sae_columns_without_filters(self):
        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        sample_rows = [
            {
                "lookup_id": "920101",
                "Tire ID": 920101,
                "Tire VDE ID": None,
                "Tire code": "QA-BASE",
                "RRC": 8.0,
                "SMERF": 6.9,
                "Reference pressure": 38.0,
                "Test load": 610.0,
                "Mileage": 1000.0,
                "alpha": -0.30,
                "beta": 1.00,
                "a": 0.0405987767,
                "b": 0.00002000,
                "c": 0.0000000500,
                "Status": "measured",
                "Source": "qa_mock_seed",
                "Notes": "Synthetic QA data",
                "_raw": {
                    "id": 920101,
                    "tire_test_code": "QA-BASE",
                    "rr_n_per_kn": 8.0,
                    "smerf": 6.9,
                    "test_pressure_value": 38.0,
                    "test_load_value": 610.0,
                    "test_mileage_km": 1000.0,
                    "sae_alpha": -0.30,
                    "sae_beta": 1.00,
                    "sae_a": 0.0405987767,
                    "sae_b": 0.00002000,
                    "sae_c": 0.0000000500,
                },
            },
            {
                "lookup_id": "920102",
                "Tire ID": 920102,
                "Tire VDE ID": None,
                "Tire code": "QA-ECO",
                "RRC": 7.0,
                "SMERF": 6.4,
                "Reference pressure": 35.0,
                "Test load": 610.0,
                "Mileage": 1000.0,
                "alpha": -0.32,
                "beta": 1.00,
                "a": 0.0388106091,
                "b": 0.00001800,
                "c": 0.0000000400,
                "Status": "measured",
                "Source": "qa_mock_seed",
                "Notes": "Synthetic QA data",
                "_raw": {
                    "id": 920102,
                    "tire_test_code": "QA-ECO",
                    "rr_n_per_kn": 7.0,
                    "smerf": 6.4,
                    "test_pressure_value": 35.0,
                    "test_load_value": 610.0,
                    "test_mileage_km": 1000.0,
                    "sae_alpha": -0.32,
                    "sae_beta": 1.00,
                    "sae_a": 0.0388106091,
                    "sae_b": 0.00001800,
                    "sae_c": 0.0000000400,
                },
            },
        ]

        with patch.object(vde_request_compact, "component_lookup_rows", return_value=sample_rows):
            self._run(app)

        self.assertFalse(app.exception)
        browser_df = next(
            item.value
            for item in app.dataframe
            if getattr(getattr(item, "value", None), "columns", None) is not None
            and {"Tire ID", "Tire code", "SMERF", "alpha", "beta", "a", "b", "c"}.issubset(set(item.value.columns))
        )
        self.assertEqual(list(browser_df["Tire code"]), ["QA-BASE", "QA-ECO"])
        self.assertEqual(list(browser_df["SMERF"]), ["6.9", "6.4"])
        self.assertEqual(app.selectbox(key="v22_lookup_selected__tire").value, "920101")

    def test_request_inputs_tire_lookup_browser_advanced_filters_remain_available(self):
        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        sample_rows = [
            {
                "lookup_id": "920101",
                "Tire ID": 920101,
                "Tire VDE ID": None,
                "Tire code": "QA-BASE",
                "RRC": 8.0,
                "Reference pressure": 32.0,
                "Test load": 500.0,
                "Mileage": 0.0,
                "alpha": -0.30,
                "beta": 1.00,
                "a": 0.0405987767,
                "b": 0.00002000,
                "c": 0.0000000500,
                "Status": "measured",
                "Source": "qa_mock_seed",
                "Notes": "Synthetic QA data",
                "_raw": {"id": 920101, "tire_test_code": "QA-BASE"},
            },
            {
                "lookup_id": "920102",
                "Tire ID": 920102,
                "Tire VDE ID": None,
                "Tire code": "QA-HIGH-PRESSURE",
                "RRC": 8.2,
                "Reference pressure": 40.0,
                "Test load": 700.0,
                "Mileage": 1000.0,
                "alpha": -0.32,
                "beta": 1.00,
                "a": 0.0388106091,
                "b": 0.00001800,
                "c": 0.0000000400,
                "Status": "measured",
                "Source": "qa_mock_seed",
                "Notes": "Synthetic QA data",
                "_raw": {"id": 920102, "tire_test_code": "QA-HIGH-PRESSURE"},
            },
        ]

        with patch.object(vde_request_compact, "component_lookup_rows", return_value=sample_rows):
            self._run(app)
            app.number_input(key="v22_tire_browser_pressure_min").set_value(39.0)
            self._run(app)

        self.assertFalse(app.exception)
        browser_df = next(
            item.value
            for item in app.dataframe
            if getattr(getattr(item, "value", None), "columns", None) is not None
            and {"Tire ID", "Tire code", "alpha", "beta", "a", "b", "c"}.issubset(set(item.value.columns))
        )
        self.assertEqual(list(browser_df["Tire code"]), ["QA-HIGH-PRESSURE"])

    def test_request_inputs_tire_lookup_browser_paginates_records(self):
        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        sample_rows = []
        for index in range(27):
            code = f"QA-{index + 1:02d}"
            sample_rows.append(
                {
                    "lookup_id": str(920100 + index),
                    "Tire ID": 920100 + index,
                    "Tire VDE ID": None,
                    "Tire code": code,
                    "RRC": 8.0 + (index * 0.01),
                    "Reference pressure": 35.0,
                    "Test load": 610.0,
                    "Mileage": 1000.0,
                    "alpha": -0.30,
                    "beta": 1.00,
                    "a": 0.04,
                    "b": 0.00002,
                    "c": 0.00000005,
                    "Status": "measured",
                    "Source": "qa_mock_seed",
                    "Notes": "Synthetic QA data",
                    "_raw": {"id": 920100 + index, "tire_test_code": code},
                }
            )

        with patch.object(vde_request_compact, "component_lookup_rows", return_value=sample_rows):
            self._run(app)
            first_page = next(
                item.value
                for item in app.dataframe
                if getattr(getattr(item, "value", None), "columns", None) is not None
                and {"Tire ID", "Tire code", "alpha", "beta", "a", "b", "c"}.issubset(set(item.value.columns))
            )
            self.assertEqual(len(first_page), 25)
            self.assertEqual(first_page.iloc[0]["Tire code"], "QA-01")
            self.assertTrue(any("Showing 1-25 of 27 records" in str(getattr(item, "value", "")) for item in app.caption))

            self._button(app.button, key="v22_tire_browser_next").click()
            self._run(app)

        self.assertFalse(app.exception)
        second_page = next(
            item.value
            for item in app.dataframe
            if getattr(getattr(item, "value", None), "columns", None) is not None
            and {"Tire ID", "Tire code", "alpha", "beta", "a", "b", "c"}.issubset(set(item.value.columns))
        )
        self.assertEqual(len(second_page), 2)
        self.assertEqual(list(second_page["Tire code"]), ["QA-26", "QA-27"])
        self.assertTrue(any("Showing 26-27 of 27 records" in str(getattr(item, "value", "")) for item in app.caption))

    def test_request_inputs_tire_lookup_uses_ctx_db_path_for_live_repository_reads(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        state["active_section"] = "inputs"
        app.session_state["ctx"] = {"db_path": str(db_path)}
        app.session_state[V22_SESSION_KEY] = state

        vde_request_compact.component_lookup_rows.clear()
        vde_request_compact.vde_lookup_rows.clear()
        self._run(app)

        self.assertFalse(app.exception)
        self.assertEqual(Path(app.session_state["_active_runtime_db_path"]).resolve(), Path(db_path).resolve())
        browser_df = next(
            item.value
            for item in app.dataframe
            if getattr(getattr(item, "value", None), "columns", None) is not None
            and {"Tire ID", "Tire code", "alpha", "beta", "a", "b", "c"}.issubset(set(item.value.columns))
        )
        self.assertIn("QA-BASE", set(browser_df["Tire code"]))
        self.assertIn("QA-ECO", set(browser_df["Tire code"]))
        self.assertIn("TIRE-QA-001", set(browser_df["Tire code"]))

    def test_request_inputs_tire_lookup_empty_and_no_match_messages_are_explicit(self):
        app = self._app()
        state = apply_v22_baseline(create_v22_state(), _baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state

        with patch.object(vde_request_compact, "component_lookup_rows", return_value=[]):
            self._run(app)
            self.assertFalse(app.exception)
            self.assertTrue(any("No Tire Database records available." in getattr(item, "value", "") for item in app.info))

        app = self._app()
        app.session_state[V22_SESSION_KEY] = state
        with patch.object(vde_request_compact, "component_lookup_rows", return_value=[]):
            self._run(app)
            app.text_input(key="v22_tire_browser_code_query").set_value("QA-ECO")
            self._run(app)
            self.assertFalse(app.exception)
            self.assertTrue(any("No matching Tire Database records." in getattr(item, "value", "") for item in app.info))

    def test_v22debug_query_param_is_ignored_after_cutover(self):
        app = self._app()
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1500.0,
                "test_mass_kg": 1500.0,
                "inertia_class": 1588.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"}],
        )
        state["active_section"] = "inputs"
        app.query_params["v22debug"] = "1"
        app.session_state[V22_SESSION_KEY] = state

        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

            self.assertFalse(app.exception)
            self.assertFalse(any("live debug is enabled" in str(item.value).lower() for item in app.warning))

            app.number_input(key="v22_simple_mass__requested_1__mass_kg").set_value(1340)
            self._button(app.button, label="Apply Mass").click()
            self._run(app)

            self.assertFalse(app.exception)
            self.assertNotIn("vde_setup_v22_debug_trace", app.session_state)

    def test_request_inputs_form_to_state_trace_mass_and_aero(self):
        app = self._app()
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1500.0,
                "test_mass_kg": 1644.0,
                "inertia_class": 1644.0,
                "cda_m2": 0.62,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC", "aero": "Absolute CdA"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "TWC shift / target class", "aero": "Absolute CdA"},
            ],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        trace = {}
        original_payload_builder = vde_request_compact.build_v22_domain_apply_payload
        original_apply_domain = vde_request_compact.apply_v22_domain_inputs

        def payload_probe(domain, proposals, values_by_proposal):
            trace[f"{domain}_form_values"] = deepcopy(values_by_proposal)
            result = original_payload_builder(domain, proposals, values_by_proposal)
            trace[f"{domain}_payload"] = deepcopy(result)
            return result

        def apply_probe(state_arg, domain, values_by_proposal):
            trace[f"{domain}_apply_input"] = deepcopy(values_by_proposal)
            result = original_apply_domain(state_arg, domain, values_by_proposal)
            proposal = next(item for item in result["proposals"] if item["proposal_id"] == "requested_1")
            trace[f"{domain}_applied_state"] = deepcopy(dict(dict(proposal.get("inputs") or {}).get(domain) or {}))
            trace[f"{domain}_domain_status"] = deepcopy(
                dict(dict(dict(result.get("domain_input_state") or {}).get(domain) or {}).get("proposal_statuses") or {}).get("requested_1")
            )
            return result

        with (
            patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()),
            patch.object(vde_request_compact, "build_v22_domain_apply_payload", side_effect=payload_probe),
            patch.object(vde_request_compact, "apply_v22_domain_inputs", side_effect=apply_probe),
        ):
            self._run(app)

            app.number_input(key="v22_simple_mass__requested_1__mass_kg").set_value(1340)
            app.selectbox(key="v22_simple_mass__requested_2__shift_steps").select("+1")
            self._button(app.button, label="Apply Mass").click()
            self._run(app)

            mass_state = app.session_state[V22_SESSION_KEY]
            mass_draft = build_v22_canonical_request_draft(mass_state)
            mass_bundle = build_v22_preview_bundle(mass_state, baseline_context=compact_baseline_context(mass_state))
            mass_req1 = mass_bundle["resolution_result"]["proposal_results"][0]
            mass_req2 = mass_bundle["resolution_result"]["proposal_results"][1]
            mass_raw = mass_draft["proposals"][0]["domain_requests"]["mass"]["raw_values"]
            mass_seed = mass_draft["proposals"][0]["domain_requests"]["mass"]["proposal_details_seed"]

            self.assertEqual(trace["mass_form_values"]["requested_1"]["mass_kg"], 1340.0)
            self.assertEqual(trace["mass_payload"]["requested_1"]["mass_kg"], 1340.0)
            self.assertEqual(trace["mass_apply_input"]["requested_1"]["mass_kg"], 1340.0)
            self.assertEqual(trace["mass_applied_state"]["mass_kg"], 1340.0)
            self.assertEqual(trace["mass_domain_status"]["status"], "applied_ready")
            self.assertEqual(mass_raw["mass_kg"], 1340.0)
            self.assertEqual(mass_seed["mass_kg"], 1340.0)
            self.assertEqual(mass_req1["domain_results"]["mass"]["requested_values"]["mass_kg"], 1340.0)
            self.assertEqual(mass_req1["resolved_snapshot"]["mass_kg"], 1340.0)
            self.assertEqual(mass_req1["resolved_snapshot"]["inertia_class"], 1474.0)
            self.assertNotIn("Curb mass is required", str(mass_req1))
            self.assertEqual(mass_req2["domain_results"]["mass"]["source"], "requested_1")
            self.assertEqual(mass_req2["resolved_snapshot"]["target_mass_kg"], 1531.0)

            self._select_request_domain(app, "aero")
            app.number_input(key="v22_simple_aero__requested_1__cda_m2").set_value(0.67)
            app.number_input(key="v22_simple_aero__requested_2__cda_m2").set_value(0.69)
            self._button(app.button, label="Apply Aero").click()
            self._run(app)

            aero_state = app.session_state[V22_SESSION_KEY]
            aero_draft = build_v22_canonical_request_draft(aero_state)
            aero_bundle = build_v22_preview_bundle(aero_state, baseline_context=compact_baseline_context(aero_state))
            aero_req1 = aero_bundle["resolution_result"]["proposal_results"][0]
            aero_req2 = aero_bundle["resolution_result"]["proposal_results"][1]
            aero_raw = aero_draft["proposals"][0]["domain_requests"]["aero"]["raw_values"]
            aero_seed = aero_draft["proposals"][0]["domain_requests"]["aero"]["proposal_details_seed"]

            self.assertEqual(trace["aero_form_values"]["requested_1"]["cda_m2"], 0.67)
            self.assertEqual(trace["aero_payload"]["requested_1"]["cda_m2"], 0.67)
            self.assertEqual(trace["aero_apply_input"]["requested_1"]["cda_m2"], 0.67)
            self.assertEqual(trace["aero_applied_state"]["cda_m2"], 0.67)
            self.assertEqual(trace["aero_domain_status"]["status"], "applied_ready")
            self.assertEqual(aero_raw["cda_m2"], 0.67)
            self.assertEqual(aero_seed["new_CdA"], 0.67)
            self.assertEqual(aero_req1["domain_results"]["aero"]["requested_values"]["new_CdA"], 0.67)
            self.assertEqual(aero_req1["resolved_snapshot"]["CdA"], 0.67)
            self.assertNotIn("New CdA is required", str(aero_req1))
            self.assertEqual(aero_req2["domain_results"]["aero"]["source"], "requested_1")
            self.assertEqual(aero_req2["domain_results"]["aero"]["requested_values"]["new_CdA"], 0.69)
            self.assertEqual(aero_req2["resolved_snapshot"]["CdA"], 0.69)

    def test_request_inputs_correction_fields_render_empty_and_preserve_zero(self):
        app = self._app()
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Custom test mass",
                    "aero": "Absolute CdA",
                    "tire": "Target final RRC",
                    "transmission": "Absolute ABC",
                    "brake": "Absolute ABC",
                }
            ],
        )
        state["active_section"] = "inputs"
        app.session_state[V22_SESSION_KEY] = state
        with patch.object(vde_request_compact, "_baseline_summary_rows", return_value=_candidate_rows()):
            self._run(app)

            self.assertIsNone(app.number_input(key="v22_correction__mass__mass_kg").value)
            self._select_request_domain(app, "aero")
            self.assertIsNone(app.number_input(key="v22_correction__aero__cda_m2").value)
            self._select_request_domain(app, "tire")
            self.assertIsNone(app.number_input(key="v22_correction__tire__rrc_N_per_kN").value)
            self._select_request_domain(app, "transmission")
            self.assertIsNone(app.number_input(key="v22_correction__transmission__trans_A_coef_N").value)
            self._select_request_domain(app, "brake")
            self.assertIsNone(app.number_input(key="v22_correction__brake__brake_A_coef_N").value)
            app.number_input(key="v22_correction__brake__brake_B_Npkph").set_value(0.0)
            app.number_input(key="v22_simple_brake__requested_1__brake_A_coef_N").set_value(2.0)
            app.number_input(key="v22_simple_brake__requested_1__brake_B_Npkph").set_value(0.0)
            app.number_input(key="v22_simple_brake__requested_1__brake_C_coef_Npkph2").set_value(0.0)
            self._button(app.button, label="Apply Brake").click()
            self._run(app)

        baseline = app.session_state[V22_SESSION_KEY]["baseline"]
        self.assertIn("brake_B_Npkph", baseline["corrections"])
        self.assertEqual(baseline["corrections"]["brake_B_Npkph"], 0.0)
        self.assertEqual(baseline["effective"]["brake_B_Npkph"], 0.0)


if __name__ == "__main__":
    unittest.main()

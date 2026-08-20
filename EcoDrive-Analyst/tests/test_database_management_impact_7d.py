from __future__ import annotations

from copy import deepcopy
import gc
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.component_repositories import load_component_repository
from src.vde_core.data_change_log_repository import fetch_change_log
from src.vde_core.database_management_contract import ChangeCommand
from src.vde_core.database_management_impact_service import (
    apply_change_with_impact,
    apply_vde_dependency_resolution,
    discover_catalog_usage,
    discover_vde_dependencies,
    preview_dependency_impact,
)
from src.vde_core.database_management_service import get_record, preview_change
from src.vde_core.qa_mock_data import seed_qa_database
from src.vde_core.repositories import fetch_vde_by_id
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_persistence import (
    REQUEST_HISTORY_PROPOSAL_TABLE,
    REQUEST_HISTORY_TABLE,
    load_v22_saved_request,
    save_v22_request,
)
from src.vde_core.vde_request_compact_state import (
    apply_v22_baseline,
    apply_v22_domain_inputs,
    apply_v22_proposal_matrix,
    create_v22_state,
)


class DatabaseManagementImpact7DTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary_directory.name) / "database_management_impact_7d.db"
        seed_qa_database(self.db_path, overwrite=False)
        self._path_context = db_module.using_db_path(self.db_path)
        self._path_context.__enter__()

    def tearDown(self):
        self._path_context.__exit__(None, None, None)
        gc.collect()
        self._temporary_directory.cleanup()

    def test_component_used_by_seven_vdes_previews_all_without_writes(self):
        save_results = [self._save_component_request(proposal_count=1) for _ in range(7)]
        component = self._component_record("transmission", "TRANS-MOCK-001")
        before_vdes = [self._vde_snapshots(item) for item in save_results]
        before_history_count = self._history_count()

        change = self._component_change(component, equivalent_A_N=float(component["equivalent_A_N"]) + 2.0)
        impact = preview_dependency_impact(change, "RECALCULATE_UPDATE", component_domain="transmission")

        self.assertTrue(impact.can_commit)
        self.assertEqual(len(impact.affected_vde_ids), 7)
        self.assertEqual(len(impact.request_recalculations), 7)
        self.assertEqual([self._vde_snapshots(item) for item in save_results], before_vdes)
        self.assertEqual(self._history_count(), before_history_count)
        self.assertEqual(get_record("COMPONENT", component["id"], component_domain="transmission")["equivalent_A_N"], component["equivalent_A_N"])

    def test_recalculate_update_preserves_ids_walk_from_and_marks_fuel_stale(self):
        save_result = self._save_component_request(proposal_count=2, walk_from=True)
        saved = {row["proposal_id"]: int(row["vde_row_id"]) for row in save_result["saved_proposals"]}
        with db_module._con() as con:
            con.execute(
                "INSERT INTO fuelcons_db (vde_id, electrification, record_origin, review_status) VALUES (?, 'ICE', 'ESTIMATED', 'CURRENT')",
                (saved["requested_2"],),
            )
            fuel_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])

        component = self._component_record("transmission", "TRANS-MOCK-001")
        change = self._component_change(component, equivalent_A_N=float(component["equivalent_A_N"]) + 3.0)
        impact = preview_dependency_impact(change, "RECALCULATE_UPDATE", component_domain="transmission")
        result = apply_change_with_impact(
            change,
            impact,
            reason="Correct transmission fixture",
            component_domain="transmission",
        )

        self.assertTrue(result.committed)
        self.assertEqual(set(result.stale_fuel_row_ids), {fuel_id})
        self.assertEqual(len(result.request_history_ids), 1)
        change_log = fetch_change_log(change.operation_id)
        self.assertEqual(change_log["impact_json"]["persistence_choice"], "RECALCULATE_UPDATE")
        self.assertEqual(len(change_log["impact_json"]["request_recalculations"]), 1)
        self.assertEqual(
            change_log["impact_json"]["request_recalculations"][0]["comparisons"][1]["proposal_id"],
            "requested_2",
        )
        latest = load_v22_saved_request(result.request_history_ids[0])
        latest_ids = {row["proposal_id"]: int(row["saved_vde_row_id"]) for row in latest["proposal_records"]}
        self.assertEqual(latest_ids, saved)
        self.assertEqual(latest["state"]["proposals"][1]["walk_from"], "requested_1")
        req1 = latest["proposal_records"][0]
        req2 = latest["proposal_records"][1]
        req1_snapshot = next(action for action in req1["component_actions"] if action.get("domain") == "transmission")["component_snapshot"]
        self.assertAlmostEqual(req1_snapshot["equivalent_A_N"], float(component["equivalent_A_N"]) + 3.0)
        self.assertNotEqual(req2["abc_total"], self._original_proposal(save_result["record_id"], "requested_2")["abc_total"])
        with db_module._con() as con:
            fuel_status = con.execute("SELECT review_status FROM fuelcons_db WHERE id=?", (fuel_id,)).fetchone()[0]
        self.assertEqual(fuel_status, "STALE_VDE")

    def test_recalculate_new_preserves_old_vdes_and_creates_new_history(self):
        save_result = self._save_component_request(proposal_count=2, walk_from=True)
        old_ids = {int(row["vde_row_id"]) for row in save_result["saved_proposals"]}
        old_rows = self._vde_snapshots(save_result)
        component = self._component_record("transmission", "TRANS-MOCK-001")
        change = self._component_change(component, equivalent_B_N_per_kph=float(component["equivalent_B_N_per_kph"]) + 0.001)
        impact = preview_dependency_impact(change, "RECALCULATE_NEW", component_domain="transmission")

        result = apply_change_with_impact(
            change,
            impact,
            reason="Correct transmission B fixture",
            component_domain="transmission",
        )
        latest = load_v22_saved_request(result.request_history_ids[0])
        new_ids = {int(row["saved_vde_row_id"]) for row in latest["proposal_records"]}

        self.assertTrue(old_ids.isdisjoint(new_ids))
        self.assertEqual(self._vde_snapshots(save_result), old_rows)
        self.assertEqual(latest["state"]["proposals"][1]["walk_from"], "requested_1")
        self.assertEqual(result.stale_fuel_row_ids, ())
        usage = discover_catalog_usage("COMPONENT", component["id"], component_domain="transmission")
        self.assertEqual(set(usage["affected_vde_ids"]), new_ids)
        self.assertEqual(
            {int(row["saved_vde_row_id"]) for row in usage["historical_usages"]},
            old_ids,
        )
        self.assertEqual(
            {row["relation"] for row in usage["historical_usages"]},
            {"historical_snapshot_direct", "historical_snapshot_walk_from"},
        )

    def test_keep_existing_changes_catalog_only_and_preserves_saved_snapshot(self):
        save_result = self._save_component_request(proposal_count=1)
        before = self._original_proposal(save_result["record_id"], "requested_1")
        history_count = self._history_count()
        component = self._component_record("transmission", "TRANS-MOCK-001")
        change = self._component_change(component, equivalent_C_N_per_kph2=float(component["equivalent_C_N_per_kph2"]) + 0.0001)
        impact = preview_dependency_impact(change, "KEEP_EXISTING", component_domain="transmission")

        result = apply_change_with_impact(
            change,
            impact,
            reason="Correct catalog while retaining historical scenarios",
            component_domain="transmission",
        )
        after = self._original_proposal(save_result["record_id"], "requested_1")

        self.assertTrue(result.committed)
        self.assertEqual(result.request_history_ids, ())
        self.assertEqual(self._history_count(), history_count)
        self.assertEqual(before["resolved_snapshot"], after["resolved_snapshot"])

    def test_tire_replacement_uses_updated_snapshot_and_canonical_resolver(self):
        save_result = self._save_tire_request()
        tire = get_record("TIRE", 920104)
        change = preview_change(
            ChangeCommand(
                entity_type="TIRE",
                action="UPDATE",
                record_id=tire["id"],
                record_origin=tire["record_origin"],
                current_record=tire,
                payload={"sae_a": float(tire["sae_a"]) + 0.003},
                reason="Correct SAE source coefficient",
            )
        )
        impact = preview_dependency_impact(change, "RECALCULATE_UPDATE")
        before = self._original_proposal(save_result["record_id"], "requested_1")

        result = apply_change_with_impact(change, impact, reason="Correct SAE source coefficient")
        latest = load_v22_saved_request(result.request_history_ids[0])
        after = latest["proposal_records"][0]
        action = next(item for item in after["component_actions"] if item.get("domain") == "tire")

        self.assertAlmostEqual(action["component_snapshot"]["tire_snapshot"]["sae_a"], float(tire["sae_a"]) + 0.003)
        self.assertNotEqual(before["abc_total"], after["abc_total"])
        self.assertNotEqual(before["vde_results"], after["vde_results"])

    def test_failure_in_middle_rolls_back_catalog_vdes_history_and_fuel(self):
        first = self._save_component_request(proposal_count=1)
        second = self._save_component_request(proposal_count=1)
        component = self._component_record("transmission", "TRANS-MOCK-001")
        original_a = component["equivalent_A_N"]
        before_first = self._vde_snapshots(first)
        before_second = self._vde_snapshots(second)
        before_history = self._history_count()
        change = self._component_change(component, equivalent_A_N=float(original_a) + 4.0)
        impact = preview_dependency_impact(change, "RECALCULATE_UPDATE", component_domain="transmission")

        def fail_after_first(index, _persisted):
            if index == 0:
                raise RuntimeError("Injected batch failure")

        with self.assertRaisesRegex(RuntimeError, "Injected batch failure"):
            apply_change_with_impact(
                change,
                impact,
                reason="Rollback fixture",
                component_domain="transmission",
                failure_injector=fail_after_first,
            )

        self.assertEqual(self._history_count(), before_history)
        self.assertEqual(self._vde_snapshots(first), before_first)
        self.assertEqual(self._vde_snapshots(second), before_second)
        self.assertEqual(get_record("COMPONENT", component["id"], component_domain="transmission")["equivalent_A_N"], original_a)

    def test_commit_rejects_dependencies_added_after_impact_preview(self):
        self._save_component_request(proposal_count=1)
        component = self._component_record("transmission", "TRANS-MOCK-001")
        original_a = component["equivalent_A_N"]
        change = self._component_change(component, equivalent_A_N=float(original_a) + 1.0)
        impact = preview_dependency_impact(change, "RECALCULATE_UPDATE", component_domain="transmission")
        self._save_component_request(proposal_count=1)
        history_count = self._history_count()

        with self.assertRaisesRegex(ValueError, "dependencies changed"):
            apply_change_with_impact(
                change,
                impact,
                reason="Stale impact preview fixture",
                component_domain="transmission",
            )

        self.assertEqual(self._history_count(), history_count)
        self.assertEqual(
            get_record("COMPONENT", component["id"], component_domain="transmission")["equivalent_A_N"],
            original_a,
        )

    def test_usage_discovery_marks_walk_from_downstream(self):
        self._save_component_request(proposal_count=2, walk_from=True)
        component = self._component_record("transmission", "TRANS-MOCK-001")
        usage = discover_catalog_usage("COMPONENT", component["id"], component_domain="transmission")

        relations = {row["proposal_id"]: row["relation"] for row in usage["usages"]}
        self.assertEqual(relations["requested_1"], "direct")
        self.assertEqual(relations["requested_2"], "walk_from_downstream")

    def test_vde_dependency_discovery_includes_fuel_and_saved_history(self):
        save_result = self._save_component_request(proposal_count=1)
        vde_id = int(save_result["saved_proposals"][0]["vde_row_id"])
        with db_module._con() as con:
            con.execute(
                "INSERT INTO fuelcons_db (vde_id, electrification, record_origin, review_status) "
                "VALUES (?, 'ICE', 'MEASURED', 'CURRENT')",
                (vde_id,),
            )

        dependencies = discover_vde_dependencies(vde_id)

        self.assertEqual(len(dependencies["fuel_rows"]), 1)
        self.assertEqual(dependencies["fuel_rows"][0]["record_origin"], "MEASURED")
        self.assertEqual(len(dependencies["saved_proposals"]), 1)
        self.assertEqual(dependencies["saved_proposals"][0]["saved_vde_row_id"], vde_id)

    def test_vde_fuel_reassignment_blocks_calculated_rows_and_rolls_back(self):
        save_result = self._save_component_request(proposal_count=1)
        vde_id = int(save_result["saved_proposals"][0]["vde_row_id"])
        with db_module._con() as con:
            con.execute(
                "INSERT INTO fuelcons_db (vde_id, electrification, record_origin, review_status) "
                "VALUES (?, 'ICE', 'MEASURED', 'CURRENT')",
                (vde_id,),
            )
            measured_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
            con.execute(
                "INSERT INTO fuelcons_db (vde_id, electrification, record_origin, review_status) "
                "VALUES (?, 'ICE', 'ESTIMATED', 'CURRENT')",
                (vde_id,),
            )
            estimated_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        preview = self._vde_change(vde_id, "REASSIGN_RELATIONSHIP")

        with self.assertRaisesRegex(ValueError, "require recalculation"):
            apply_vde_dependency_resolution(
                preview,
                resolution_action="REASSIGN_FUEL",
                fuel_row_ids=(measured_id, estimated_id),
                replacement_vde_id=900002,
                reason="Move source Fuel rows",
            )
        with db_module._con() as con:
            after_failure = con.execute(
                "SELECT COUNT(*), MIN(vde_id), MAX(vde_id) FROM fuelcons_db WHERE id IN (?, ?)",
                (measured_id, estimated_id),
            ).fetchone()
        self.assertEqual(tuple(after_failure), (2, vde_id, vde_id))

        result = apply_vde_dependency_resolution(
            preview,
            resolution_action="REASSIGN_FUEL",
            fuel_row_ids=(measured_id,),
            replacement_vde_id=900002,
            reason="Move measured Fuel row",
        )

        self.assertTrue(result.committed)
        with db_module._con() as con:
            measured = con.execute(
                "SELECT vde_id, review_status FROM fuelcons_db WHERE id=?", (measured_id,)
            ).fetchone()
            estimated_vde = con.execute("SELECT vde_id FROM fuelcons_db WHERE id=?", (estimated_id,)).fetchone()[0]
        self.assertEqual(tuple(measured), (900002, "REVIEW_REQUIRED"))
        self.assertEqual(estimated_vde, vde_id)

    def test_vde_delete_requires_explicit_fuel_selection_and_preserves_history(self):
        save_result = self._save_component_request(proposal_count=1)
        vde_id = int(save_result["saved_proposals"][0]["vde_row_id"])
        with db_module._con() as con:
            fuel_ids = []
            for origin in ("MEASURED", "ESTIMATED"):
                con.execute(
                    "INSERT INTO fuelcons_db (vde_id, electrification, record_origin, review_status) "
                    "VALUES (?, 'ICE', ?, 'CURRENT')",
                    (vde_id, origin),
                )
                fuel_ids.append(int(con.execute("SELECT last_insert_rowid()").fetchone()[0]))
        preview = self._vde_change(vde_id, "DELETE")

        with self.assertRaisesRegex(ValueError, "Every linked Fuel row"):
            apply_vde_dependency_resolution(
                preview,
                resolution_action="DELETE_FUEL_AND_VDE",
                fuel_row_ids=(fuel_ids[0],),
                reason="Incomplete dependency selection",
            )
        with db_module._con() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM vde_db WHERE id=?", (vde_id,)).fetchone()[0], 1)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM fuelcons_db WHERE vde_id=?", (vde_id,)).fetchone()[0], 2)

        result = apply_vde_dependency_resolution(
            preview,
            resolution_action="DELETE_FUEL_AND_VDE",
            fuel_row_ids=tuple(fuel_ids),
            reason="Delete explicit Fuel dependencies and VDE",
        )

        self.assertTrue(result.committed)
        with db_module._con() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM vde_db WHERE id=?", (vde_id,)).fetchone()[0], 0)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM fuelcons_db WHERE vde_id=?", (vde_id,)).fetchone()[0], 0)
            self.assertEqual(
                con.execute(
                    f"SELECT COUNT(*) FROM {REQUEST_HISTORY_PROPOSAL_TABLE} WHERE saved_vde_row_id=?",
                    (vde_id,),
                ).fetchone()[0],
                1,
            )

    def _save_component_request(self, *, proposal_count: int, walk_from: bool = False) -> dict:
        baseline = fetch_vde_by_id(900001)
        component = load_component_repository("transmission").get_by_id("TRANS-MOCK-001")
        state = apply_v22_baseline(create_v22_state(), deepcopy(baseline))
        matrix = []
        inputs = {}
        for index in range(1, proposal_count + 1):
            proposal_id = f"requested_{index}"
            direct = index == 1 or not walk_from
            matrix.append(
                {
                    "proposal_id": proposal_id,
                    "walk_from": "requested_1" if walk_from and index > 1 else "baseline",
                    "transmission": "Lookup from DB" if direct else "Inherit",
                }
            )
            if direct:
                inputs[proposal_id] = {
                    "transmission_component_db_id": component["component_id"],
                    "trans_A_coef_N": component["trans_A"],
                    "trans_B_coef_Npkph": component["trans_B"],
                    "trans_C_coef_Npkph2": component["trans_C"],
                    "transmission_loss_pct": component["loss_pct"],
                }
        state = apply_v22_proposal_matrix(state, matrix)
        state = apply_v22_domain_inputs(state, "transmission", inputs)
        state["proposals"] = list(state["proposals"])[:proposal_count]
        return self._save_state(state)

    def _save_tire_request(self) -> dict:
        baseline = fetch_vde_by_id(900001)
        state = apply_v22_baseline(create_v22_state(), deepcopy(baseline))
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire DB lookup"}],
        )
        tire = get_record("TIRE", 920104)
        state = apply_v22_domain_inputs(
            state,
            "tire",
            {
                "requested_1": {
                    "tire_db_id": tire["id"],
                    "tire_code": tire["tire_test_code"],
                    "rrc_N_per_kN": tire["rr_n_per_kn"],
                    "front_pressure_psi": 30.0,
                    "rear_pressure_psi": 30.0,
                    "tire_load_mass_basis": "TEST_MASS",
                    "tire_snapshot": deepcopy(tire),
                }
            },
        )
        return self._save_state(state)

    def _save_state(self, state: dict) -> dict:
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        state["preview"] = {"status": "fresh", "fingerprint": bundle["fingerprint"], "result": bundle}
        result = save_v22_request(state)
        self.assertEqual(result["status"], "success", result)
        return result

    def _component_record(self, domain: str, code: str) -> dict:
        with db_module._con() as con:
            row_id = int(con.execute("SELECT id FROM component_db WHERE domain=? AND component_code=?", (domain, code)).fetchone()[0])
        return get_record("COMPONENT", row_id, component_domain=domain)

    def _component_change(self, component: dict, **payload) -> object:
        return preview_change(
            ChangeCommand(
                entity_type="COMPONENT",
                action="UPDATE",
                record_id=component["id"],
                record_origin=component["record_origin"],
                current_record=component,
                payload=payload,
                reason="Correct component fixture",
            )
        )

    def _vde_change(self, vde_id: int, action: str) -> object:
        record = get_record("VDE", vde_id)
        return preview_change(
            ChangeCommand(
                entity_type="VDE",
                action=action,
                record_id=vde_id,
                record_origin=record["record_origin"],
                current_record=record,
                reason="Resolve VDE dependencies",
            )
        )

    def _history_count(self) -> int:
        with db_module._con() as con:
            return int(con.execute(f"SELECT COUNT(*) FROM {REQUEST_HISTORY_TABLE}").fetchone()[0])

    def _original_proposal(self, request_history_id: int, proposal_id: str) -> dict:
        loaded = load_v22_saved_request(request_history_id)
        return next(row for row in loaded["proposal_records"] if row["proposal_id"] == proposal_id)

    def _vde_snapshots(self, save_result: dict) -> dict[int, tuple]:
        ids = [int(row["vde_row_id"]) for row in save_result["saved_proposals"]]
        with db_module._con() as con:
            return {
                row[0]: tuple(row[1:])
                for row in con.execute(
                    f"SELECT id, coast_A_N, coast_B_N_per_kph, coast_C_N_per_kph2, vde_total_mj_per_km, vde_net_mj_per_km, updated_at FROM vde_db WHERE id IN ({','.join('?' for _ in ids)}) ORDER BY id",
                    ids,
                ).fetchall()
            }


if __name__ == "__main__":
    unittest.main()

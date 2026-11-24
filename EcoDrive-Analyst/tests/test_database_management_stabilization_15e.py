from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.vde_app.components import database_management as page
from src.vde_app.database_management_stage import (
    StageDatabaseIdentityError,
    add_stage_command,
    assert_stage_database_identity,
    canonical_database_identity,
    execute_sequential_commit,
)
from src.vde_core.database_management_contract import ChangeResult


def _command(record_id: int, *, action: str = "UPDATE", **extra):
    return {
        "stage_operation_id": f"op-{record_id}",
        "entity_type": "VDE",
        "action": action,
        "record_id": record_id,
        **extra,
    }


def _result(record_id: int):
    return ChangeResult(
        f"op-{record_id}", "VDE", "UPDATE", True,
        change_log_id=record_id + 10, affected_record_ids=(str(record_id),),
    )


class StageDatabaseIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_a = Path(self.temp_dir.name) / "a.sqlite"
        self.db_b = Path(self.temp_dir.name) / "b.sqlite"
        for path in (self.db_a, self.db_b):
            connection = sqlite3.connect(path)
            try:
                connection.execute("CREATE TABLE writes (value INTEGER)")
                connection.commit()
            finally:
                connection.close()

    def test_stg_01_stage_is_bound_to_active_database(self):
        commands, origin = add_stage_command(
            [], stage_database_identity=None, active_database_identity=str(self.db_a), command=_command(1)
        )
        self.assertEqual(origin, canonical_database_identity(self.db_a))
        self.assertEqual(
            assert_stage_database_identity(
                commands=commands, stage_database_identity=origin, active_database_identity=str(self.db_a)
            ),
            origin,
        )

    def test_stg_02_preview_and_commit_are_blocked_on_database_b(self):
        commands, origin = add_stage_command(
            [], stage_database_identity=None, active_database_identity=str(self.db_a), command=_command(1)
        )
        with self.assertRaises(StageDatabaseIdentityError):
            assert_stage_database_identity(
                commands=commands, stage_database_identity=origin, active_database_identity=str(self.db_b)
            )

    def test_stg_03_returning_to_database_a_recovers_stage(self):
        commands, origin = add_stage_command(
            [], stage_database_identity=None, active_database_identity=str(self.db_a), command=_command(1)
        )
        with self.assertRaises(StageDatabaseIdentityError):
            assert_stage_database_identity(
                commands=commands, stage_database_identity=origin, active_database_identity=str(self.db_b)
            )
        self.assertEqual(
            assert_stage_database_identity(
                commands=commands, stage_database_identity=origin, active_database_identity=str(self.db_a)
            ),
            origin,
        )

    def test_stg_04_cross_database_stage_mixing_is_refused(self):
        commands, origin = add_stage_command(
            [], stage_database_identity=None, active_database_identity=str(self.db_a), command=_command(1)
        )
        with self.assertRaises(StageDatabaseIdentityError):
            add_stage_command(
                commands, stage_database_identity=origin,
                active_database_identity=str(self.db_b), command=_command(2)
            )

    def test_stg_05_explicit_discard_changes_no_database(self):
        before = {path: path.read_bytes() for path in (self.db_a, self.db_b)}
        streamlit = SimpleNamespace(session_state={
            page.STAGE_KEY: [_command(1)],
            page.STAGE_DB_IDENTITY_KEY: canonical_database_identity(self.db_a),
            page.COMMIT_RECONCILIATION_KEY: True,
        })
        with patch.object(page, "st", streamlit):
            page._clear_stage()
        self.assertEqual(streamlit.session_state[page.STAGE_KEY], [])
        self.assertEqual(before, {path: path.read_bytes() for path in (self.db_a, self.db_b)})

    def test_stg_06_import_rows_obey_same_cross_database_guard(self):
        row = _command(1, action="CREATE", import_batch_id="batch", import_sheet="VDE", import_row=2)
        commands, origin = add_stage_command(
            [], stage_database_identity=None, active_database_identity=str(self.db_a), command=row
        )
        with self.assertRaises(StageDatabaseIdentityError):
            add_stage_command(
                commands, stage_database_identity=origin, active_database_identity=str(self.db_b),
                command=_command(2, action="CREATE", import_batch_id="batch", import_sheet="VDE", import_row=3),
            )

    def test_stg_07_relative_and_absolute_paths_have_one_identity(self):
        current = Path.cwd()
        try:
            os.chdir(self.temp_dir.name)
            relative = canonical_database_identity("a.sqlite")
        finally:
            os.chdir(current)
        self.assertEqual(relative, canonical_database_identity(self.db_a))
        self.assertEqual(
            canonical_database_identity(str(self.db_a).upper()),
            canonical_database_identity(self.db_a),
        )
        alias = Path(self.temp_dir.name) / "alias.sqlite"
        try:
            alias.symlink_to(self.db_a)
        except OSError:
            pass  # Windows may require Developer Mode for symlink creation.
        else:
            self.assertEqual(canonical_database_identity(alias), canonical_database_identity(self.db_a))

    def test_stg_08_mismatch_performs_zero_writes_and_zero_receipts(self):
        applied: list[int] = []
        receipts: list[str] = []
        outcome = execute_sequential_commit(
            [_command(1)],
            stage_database_identity=canonical_database_identity(self.db_a),
            active_database_identity=lambda: str(self.db_b),
            apply_one=lambda *_args: applied.append(1),
            receipt_lookup=lambda operation_id: receipts.append(operation_id),
        )
        self.assertEqual(outcome.status, "BLOCKED")
        self.assertEqual(applied, [])
        self.assertEqual(receipts, [])
        connection = sqlite3.connect(self.db_b)
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM writes").fetchone()[0], 0)
        finally:
            connection.close()


class SequentialCommitTests(unittest.TestCase):
    def test_receipt_lookup_error_blocks_before_any_write(self):
        applied: list[int] = []
        outcome = execute_sequential_commit(
            [_command(1)], stage_database_identity="db.sqlite",
            active_database_identity=lambda: "db.sqlite",
            apply_one=lambda _index, command: applied.append(command["record_id"]) or _result(1),
            receipt_lookup=lambda _operation_id: (_ for _ in ()).throw(sqlite3.OperationalError("receipt DB")),
        )
        self.assertEqual(outcome.status, "BLOCKED")
        self.assertTrue(outcome.uncertain)
        self.assertEqual(applied, [])

    def test_cmt_01_two_updates_complete_with_receipts(self):
        outcome = execute_sequential_commit(
            [_command(1), _command(2)], stage_database_identity="db.sqlite",
            active_database_identity=lambda: "db.sqlite",
            apply_one=lambda _index, command: _result(command["record_id"]),
            receipt_lookup=lambda _operation_id: None,
        )
        self.assertEqual(outcome.status, "COMPLETE")
        self.assertEqual(outcome.committed_count, 2)
        self.assertEqual([entry.change_log_id for entry in outcome.entries], [11, 12])
        self.assertEqual(outcome.remaining_commands, ())

    def test_cmt_02_retry_does_not_replay_confirmed_create(self):
        calls: list[int] = []

        def first_apply(_index, command):
            calls.append(command["record_id"])
            if command["record_id"] == 2:
                raise sqlite3.OperationalError("injected")
            return _result(command["record_id"])

        first = execute_sequential_commit(
            [_command(1, action="CREATE"), _command(2, action="CREATE")],
            stage_database_identity="db.sqlite", active_database_identity=lambda: "db.sqlite",
            apply_one=first_apply, receipt_lookup=lambda _operation_id: None,
        )
        second = execute_sequential_commit(
            first.remaining_commands, stage_database_identity="db.sqlite",
            active_database_identity=lambda: "db.sqlite",
            apply_one=lambda _index, command: calls.append(command["record_id"]) or _result(command["record_id"]),
            receipt_lookup=lambda _operation_id: None,
        )
        self.assertEqual(calls, [1, 2, 2])
        self.assertEqual(second.status, "COMPLETE")

    def test_cmt_03_impact_success_is_removed_before_later_failure(self):
        outcome = execute_sequential_commit(
            [_command(1, physics_action="RECALCULATE_UPDATE"), _command(2)],
            stage_database_identity="db.sqlite", active_database_identity=lambda: "db.sqlite",
            apply_one=lambda _index, command: _result(1) if command["record_id"] == 1
            else (_ for _ in ()).throw(ValueError("later failure")),
            receipt_lookup=lambda _operation_id: None,
        )
        self.assertEqual(outcome.committed_count, 1)
        self.assertEqual([row["record_id"] for row in outcome.remaining_commands], [2])

    def test_cmt_04_database_change_between_operations_stops_before_write(self):
        identities = iter(("db-a.sqlite", "db-b.sqlite"))
        applied: list[int] = []
        outcome = execute_sequential_commit(
            [_command(1), _command(2)], stage_database_identity="db-a.sqlite",
            active_database_identity=lambda: next(identities),
            apply_one=lambda _index, command: applied.append(command["record_id"]) or _result(command["record_id"]),
            receipt_lookup=lambda _operation_id: None,
        )
        self.assertEqual(applied, [1])
        self.assertEqual(outcome.status, "PARTIAL")
        self.assertEqual([row["record_id"] for row in outcome.remaining_commands], [2])

    def test_cmt_06_import_failure_reports_confirmed_and_pending_rows(self):
        commands = [
            _command(row, action="CREATE", import_batch_id="batch", import_sheet="VDE", import_row=row,
                     import_filename="import.xlsx", import_counts={"skipped": 1, "invalid": 2})
            for row in (1, 2, 3)
        ]
        outcome = execute_sequential_commit(
            commands, stage_database_identity="db.sqlite", active_database_identity=lambda: "db.sqlite",
            apply_one=lambda _index, command: _result(1) if command["record_id"] == 1
            else (_ for _ in ()).throw(sqlite3.OperationalError("row 2")),
            receipt_lookup=lambda _operation_id: None,
        )
        streamlit = SimpleNamespace(session_state={})
        with patch.object(page, "st", streamlit):
            page._store_import_commit_report(commands, outcome)
        report = streamlit.session_state[page.IMPORT_REPORT_KEY][0]
        self.assertEqual((report["inserted"], report["pending"]), (1, 2))
        self.assertIn("Partial", report["status"])

    def test_cmt_07_discard_after_partial_does_not_undo_confirmed_write(self):
        persisted: list[int] = []

        def apply_one(_index, command):
            if command["record_id"] == 2:
                raise ValueError("stop")
            persisted.append(command["record_id"])
            return _result(command["record_id"])

        outcome = execute_sequential_commit(
            [_command(1), _command(2)], stage_database_identity="db.sqlite",
            active_database_identity=lambda: "db.sqlite",
            apply_one=apply_one,
            receipt_lookup=lambda _operation_id: None,
        )
        streamlit = SimpleNamespace(session_state={
            page.STAGE_KEY: list(outcome.remaining_commands), page.STAGE_DB_IDENTITY_KEY: "db.sqlite"
        })
        with patch.object(page, "st", streamlit):
            page._clear_stage()
        self.assertEqual(persisted, [1])
        self.assertEqual(streamlit.session_state[page.STAGE_KEY], [])

    def test_cmt_08_dependency_failure_leaves_operation_pending(self):
        outcome = execute_sequential_commit(
            [_command(1, resolution_action="DELETE_FUEL_AND_VDE")],
            stage_database_identity="db.sqlite", active_database_identity=lambda: "db.sqlite",
            apply_one=lambda *_args: (_ for _ in ()).throw(sqlite3.IntegrityError("dependency")),
            receipt_lookup=lambda _operation_id: None,
        )
        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.committed_count, 0)
        self.assertEqual(len(outcome.remaining_commands), 1)


class _Column:
    def __init__(self, *, click_commit: bool = False):
        self.click_commit = click_commit

    def button(self, label, **_kwargs):
        return self.click_commit and label == "Commit reviewed changes"

    def caption(self, *_args, **_kwargs):
        pass


class _StreamlitStub:
    def __init__(self, staged, database_identity):
        self.session_state = {
            page.STAGE_KEY: list(staged), page.STAGE_DB_IDENTITY_KEY: database_identity,
        }
        self.errors: list[str] = []
        self.successes: list[str] = []

    def subheader(self, *_args, **_kwargs):
        pass

    def caption(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def text_area(self, *_args, **_kwargs):
        return "Regression test"

    def checkbox(self, *_args, **_kwargs):
        return False

    def dataframe(self, *_args, **_kwargs):
        pass

    def columns(self, *_args, **_kwargs):
        return [_Column(), _Column(click_commit=True), _Column()]

    def error(self, message, **_kwargs):
        self.errors.append(str(message))

    def success(self, message, **_kwargs):
        self.successes.append(str(message))

    def rerun(self):
        pass


def _preview(record_id: int):
    return SimpleNamespace(
        operation_id=f"op-{record_id}", action="UPDATE", entity_type="VDE", record_id=str(record_id),
        field_diff=(), validation_issues=(), requires_reason=True, can_commit=True,
    )


class DatabaseManagementPageRecoveryTests(unittest.TestCase):
    def test_cmt_05_operational_error_removes_only_confirmed_successes(self):
        database_identity = canonical_database_identity("db.sqlite")
        staged = [_command(record_id) for record_id in (1, 2, 3)]
        streamlit = _StreamlitStub(staged, database_identity)
        with (
            patch.object(page, "st", streamlit),
            patch.object(page, "_active_database_identity", return_value=database_identity),
            patch.object(page, "_preview_for_stage", side_effect=[_preview(1), _preview(2), _preview(3)]),
            patch.object(page, "_render_impact_preview", return_value=None),
            patch.object(page, "apply_change", side_effect=[
                _result(1), sqlite3.OperationalError("injected second operation failure")
            ]),
            patch.object(page, "fetch_change_log", return_value=None),
        ):
            page._render_review_changes()

        self.assertEqual(
            [command["record_id"] for command in streamlit.session_state[page.STAGE_KEY]], [2, 3]
        )
        self.assertTrue(any("1 applied" in message for message in streamlit.errors))


if __name__ == "__main__":
    unittest.main()

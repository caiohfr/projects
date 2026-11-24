from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Methodology_Models.py"


class MethodologyModelsUiTests(unittest.TestCase):
    def _app(self) -> AppTest:
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_page_exposes_seven_topic_internal_index(self):
        app = self._app()
        self.assertTrue(any(item.value == "Methodology & Models" for item in app.title))
        selector = app.radio(key="methodology_topic")
        self.assertEqual(
            selector.options,
            [
                "Vehicle Demanded Energy",
                "Roadload & Coastdown",
                "Powertrain L0",
                "Tires & Components",
                "Contracts / TOTAL vs NET",
                "Experimental Methods",
                "References & Provenance",
            ],
        )

    def test_total_net_topic_uses_current_contract_and_removes_obsolete_direction(self):
        app = self._app()
        app.radio(key="methodology_topic").set_value("Contracts / TOTAL vs NET").run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        surface = "\n".join(str(item.value) for item in [*app.markdown, *app.table])
        self.assertIn("authoritative coastdown", surface.lower())
        self.assertIn("Never substituted with TOTAL", surface)
        self.assertNotIn("VDE_NET represents the ideal tractive energy", surface)
        self.assertNotIn("Add Transmission Losses?", surface)

    def test_experimental_topic_does_not_publish_future_capabilities(self):
        app = self._app()
        app.radio(key="methodology_topic").set_value("Experimental Methods").run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        surface = "\n".join(str(item.value) for item in [*app.markdown, *app.table])
        self.assertIn("Planned", surface)
        self.assertIn("Unavailable as a published workflow", surface)


if __name__ == "__main__":
    unittest.main()

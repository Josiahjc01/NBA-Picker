"""Offline learning store smoke tests (no network)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data import learning_store as ls


class LearningStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.learn = self.root / "learning"
        self.out = self.root / "output"
        self.out.mkdir()
        self.learn.mkdir()
        self._prev_learn = config.LEARNING_DIR
        self._prev_out = config.OUTPUT_DIR
        self._prev_cache = config.CACHE_DIR
        config.LEARNING_DIR = self.learn
        config.OUTPUT_DIR = self.out
        config.CACHE_DIR = self.root
        # Reset runtime knobs to import defaults
        for k, v in ls.config_defaults().items():
            setattr(config, k, float(v))

    def tearDown(self) -> None:
        config.LEARNING_DIR = self._prev_learn
        config.OUTPUT_DIR = self._prev_out
        config.CACHE_DIR = self._prev_cache
        ls.apply_learning_knobs()
        self._tmp.cleanup()

    def test_ascii_paste_parse(self) -> None:
        text = "\n".join(
            [
                "DATE 2026-11-01",
                "BOS @ ORL",
                "ACT 109-107",
                "PRED 117.7-109.8 P(home)=0.297",
            ]
        )
        obj = ls.parse_ascii_paste(text)
        assert obj is not None
        self.assertEqual(obj["kind"], "real_box")
        self.assertEqual(obj["away_abbr"], "BOS")
        self.assertEqual(obj["home_abbr"], "ORL")
        self.assertEqual(obj["act_away_pts"], 109.0)
        self.assertEqual(obj["paired_prediction"]["pred_away_pts"], 117.7)

    def test_import_paste_grades_means_only(self) -> None:
        paste = {
            "schema_version": 1,
            "kind": "ai_comparison",
            "date": "2026-11-01",
            "away_abbr": "BOS",
            "home_abbr": "ORL",
            "act_away_pts": 109,
            "act_home_pts": 107,
            "pred_away_pts": 117.7,
            "pred_home_pts": 109.8,
            "pred_home_win_prob": 0.297,
            "ai_summary": "test",
            "suggested_knobs": {"SCORE_SIGMA": 12.0},
            "suggested_code_themes": ["injury_usage_spillover_weights"],
            "confidence": 0.9,
            # Poison field — must never be used for grading
            "highlight_sim_max_pts": 55,
        }
        result = ls.import_paste(paste, auto_knobs=False)
        self.assertTrue(result["ok"])
        grade = result["grade"]
        self.assertEqual(grade["pred_away_pts"], 117.7)
        self.assertEqual(grade["act_away_pts"], 109)
        self.assertIn("abs_err_total", grade)
        self.assertTrue((self.learn / "paste_inbox").exists())
        self.assertTrue(any((self.learn / "paste_inbox").glob("*.json")))
        # Code themes inbox only — never applied as code
        self.assertIsNotNone(result.get("code_themes_inbox"))
        inbox = list((self.learn / "suggestions_inbox").glob("*.json"))
        self.assertEqual(len(inbox), 1)
        packed = json.loads(inbox[0].read_text(encoding="utf-8"))
        self.assertEqual(packed["kind"], "code_change")
        self.assertIn("injury_usage_spillover_weights", packed["suggested_code_themes"])

    def test_knobs_apply_and_reset(self) -> None:
        defaults = ls.config_defaults()
        ls.save_knobs(
            {**defaults, "SCORE_SIGMA": 12.5, "HOME_COURT_POINTS": 3.2},
            n_games_basis=25,
            reason="test",
        )
        knobs = ls.apply_learning_knobs()
        self.assertAlmostEqual(float(config.SCORE_SIGMA), 12.5, places=4)
        self.assertAlmostEqual(float(config.HOME_COURT_POINTS), 3.2, places=4)
        self.assertIn("SCORE_SIGMA", knobs)
        ls.reset_knobs_to_defaults(reason="test_reset")
        self.assertAlmostEqual(
            float(config.SCORE_SIGMA), defaults["SCORE_SIGMA"], places=4
        )
        hist = (self.learn / "knob_history.jsonl").read_text(encoding="utf-8")
        self.assertIn("reset", hist)

    def test_step_and_caps(self) -> None:
        self.assertAlmostEqual(ls.step_toward(10.0, 20.0, 0.10), 11.0, places=6)
        capped = ls._clip_knob("HOME_COURT_POINTS", 9.0)
        self.assertLessEqual(capped, 4.0)
        capped_lo = ls._clip_knob("REST_B2B_POINTS", -9.0)
        self.assertGreaterEqual(capped_lo, -3.0)

    def test_auto_knobs_min_games(self) -> None:
        # Seed fewer than min games
        for i in range(5):
            grade = ls.build_grade_row(
                game_date="2026-11-01",
                away_abbr="BOS",
                home_abbr="ORL",
                pred_away_pts=110 + i,
                pred_home_pts=108,
                pred_home_win_prob=0.45,
                act_away_pts=109,
                act_home_pts=107,
                source="test",
            )
            ls.append_graded_game(grade)
        result = ls.auto_update_knobs(reason="test")
        self.assertFalse(result["applied"])
        self.assertIn("Need >=", result.get("skipped_reason") or "")

    def test_auto_knobs_steps_after_min(self) -> None:
        # Seed enough games with systematic total miss
        for i in range(config.LEARNING_MIN_GAMES):
            grade = ls.build_grade_row(
                game_date="2026-11-01",
                away_abbr="BOS",
                home_abbr="ORL",
                pred_away_pts=120.0,
                pred_home_pts=118.0,
                pred_home_win_prob=0.55,
                act_away_pts=100.0,
                act_home_pts=98.0,
                source="test",
            )
            ls.append_graded_game(grade)
        before = float(config.SCORE_SIGMA)
        result = ls.auto_update_knobs(reason="test_enough", force=True)
        self.assertTrue(result["applied"])
        self.assertIn("SCORE_SIGMA", result["changes"])
        after = float(config.SCORE_SIGMA)
        # Moved toward higher sigma (large RMSE) but only a step
        self.assertNotEqual(before, after)
        # Within hard caps
        self.assertGreaterEqual(after, 8.0)
        self.assertLessEqual(after, 14.0)


if __name__ == "__main__":
    unittest.main()

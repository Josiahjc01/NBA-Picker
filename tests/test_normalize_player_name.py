"""Accent-folding name joins (Dončić ↔ Doncic) and roster alignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.process import align_players_to_rosters, index_injuries, project_player_minutes
from safe_text import normalize_player_name


class NormalizePlayerNameTests(unittest.TestCase):
    def test_doncic_accent_fold(self) -> None:
        self.assertEqual(
            normalize_player_name("Luka Dončić"),
            normalize_player_name("Luka Doncic"),
        )
        self.assertEqual(normalize_player_name("Luka Dončić"), "luka doncic")

    def test_common_accented_nba_names(self) -> None:
        pairs = [
            ("Nikola Jokić", "Nikola Jokic"),
            ("Dario Šarić", "Dario Saric"),
            ("Juan Núñez", "Juan Nunez"),
            ("Bogdan Bogdanović", "Bogdan Bogdanovic"),
            ("Kristaps Porziņģis", "Kristaps Porzingis"),
        ]
        for accented, ascii_name in pairs:
            self.assertEqual(
                normalize_player_name(accented),
                normalize_player_name(ascii_name),
                msg=f"{accented!r} vs {ascii_name!r}",
            )

    def test_whitespace_and_empty(self) -> None:
        self.assertEqual(normalize_player_name("  Luka   Doncic "), "luka doncic")
        self.assertEqual(normalize_player_name(""), "")
        self.assertEqual(normalize_player_name(None), "")


class AlignPlayersAccentTests(unittest.TestCase):
    def test_align_keeps_doncic_season_stats_on_ascii_roster(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "PLAYER_NAME": "Luka Dončić",
                    "TEAM_ABBREVIATION": "DAL",  # stale pre-trade tag
                    "MIN": 35.7,
                    "PTS": 38.3,
                    "REB": 8.1,
                    "AST": 9.2,
                    "STL": 1.1,
                    "BLK": 0.4,
                    "GP": 7,
                },
                {
                    "PLAYER_NAME": "Austin Reaves",
                    "TEAM_ABBREVIATION": "LAL",
                    "MIN": 34.0,
                    "PTS": 26.0,
                    "REB": 4.0,
                    "AST": 5.0,
                    "STL": 1.0,
                    "BLK": 0.2,
                    "GP": 7,
                },
            ]
        )
        rosters = [
            {
                "PLAYER_NAME": "Luka Doncic",
                "TEAM_ABBREVIATION": "LAL",
                "STATUS": "active",
                "PLAYER_ID": "3945274",
            },
            {
                "PLAYER_NAME": "Austin Reaves",
                "TEAM_ABBREVIATION": "LAL",
                "STATUS": "active",
                "PLAYER_ID": "4066261",
            },
        ]
        aligned = align_players_to_rosters(players, rosters)
        self.assertEqual(len(aligned), 2)
        luka = aligned[aligned["PLAYER_NAME"].astype(str).str.contains("Luka", case=False)]
        self.assertEqual(len(luka), 1)
        row = luka.iloc[0]
        self.assertEqual(row["PLAYER_NAME"], "Luka Doncic")
        self.assertEqual(row["TEAM_ABBREVIATION"], "LAL")
        self.assertAlmostEqual(float(row["MIN"]), 35.7)
        self.assertAlmostEqual(float(row["PTS"]), 38.3)
        self.assertEqual(row["SOURCE"], "season+roster")
        self.assertEqual(str(row["PLAYER_ID"]), "3945274")
        # No silent roster_only duplicate for Doncic
        sources = aligned["SOURCE"].tolist()
        self.assertNotIn("roster_only", sources)

    def test_soft_last_name_match_uses_folded_name(self) -> None:
        # Accented season name vs ASCII roster; first-initial+last still links if
        # spelling differs slightly after fold... exact fold already works; this
        # covers soft path when middle tokens differ.
        players = pd.DataFrame(
            [
                {
                    "PLAYER_NAME": "L. Dončić",
                    "TEAM_ABBREVIATION": "LAL",
                    "MIN": 35.7,
                    "PTS": 38.3,
                    "GP": 7,
                }
            ]
        )
        rosters = [
            {
                "PLAYER_NAME": "Luka Doncic",
                "TEAM_ABBREVIATION": "LAL",
                "STATUS": "active",
                "PLAYER_ID": "3945274",
            }
        ]
        aligned = align_players_to_rosters(players, rosters)
        self.assertEqual(len(aligned), 1)
        self.assertAlmostEqual(float(aligned.iloc[0]["MIN"]), 35.7)
        self.assertEqual(aligned.iloc[0]["PLAYER_NAME"], "Luka Doncic")


class MinutesFallbackTests(unittest.TestCase):
    def test_roster_only_healthy_gets_floor_not_zero(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "PLAYER_NAME": "Luka Doncic",
                    "TEAM_ABBREVIATION": "LAL",
                    "MIN": 0.0,
                    "PTS": None,
                    "ON_ROSTER": True,
                    "ROSTER_STATUS": "active",
                    "SOURCE": "roster_only",
                }
            ]
        )
        minutes = project_player_minutes(players, injuries={})
        self.assertEqual(len(minutes), 1)
        self.assertGreaterEqual(float(minutes.iloc[0]["proj_minutes"]), 10.0)
        self.assertGreaterEqual(float(minutes.iloc[0]["adj_minutes"]), 10.0)
        self.assertEqual(minutes.iloc[0]["minutes_source"], "rotation_floor")

    def test_out_injury_still_zeros_minutes(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "PLAYER_NAME": "Luka Doncic",
                    "TEAM_ABBREVIATION": "LAL",
                    "MIN": 35.7,
                    "PTS": 38.3,
                    "ON_ROSTER": True,
                    "ROSTER_STATUS": "active",
                    "SOURCE": "season+roster",
                }
            ]
        )
        injuries = index_injuries(
            [{"player": "Luka Dončić", "team_abbr": "LAL", "status": "Out"}]
        )
        minutes = project_player_minutes(players, injuries=injuries)
        self.assertAlmostEqual(float(minutes.iloc[0]["adj_minutes"]), 0.0)
        self.assertAlmostEqual(float(minutes.iloc[0]["minutes_factor"]), 0.0)

    def test_recent_minutes_lookup_accent_folded(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "PLAYER_NAME": "Luka Doncic",
                    "TEAM_ABBREVIATION": "LAL",
                    "MIN": 0.0,
                    "ON_ROSTER": True,
                    "ROSTER_STATUS": "active",
                    "SOURCE": "roster_only",
                }
            ]
        )
        recent = {
            normalize_player_name("Luka Dončić"): {
                "recent_minutes": 36.5,
                "usage_fga36": 22.0,
            }
        }
        minutes = project_player_minutes(players, injuries={}, recent_minutes=recent)
        self.assertGreaterEqual(float(minutes.iloc[0]["proj_minutes"]), 20.0)
        self.assertEqual(minutes.iloc[0]["minutes_source"], "recent")


if __name__ == "__main__":
    unittest.main()

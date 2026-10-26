import argparse
import sqlite3
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from src.Utils.features import TEAM_FEATURES
from src.Utils.blend import blend_team_stats, load_local_prior, dataset_table, validate_settings
from src.Utils.snapshots import calendar_date, snapshot_dates, select_snapshot_date

from src.Utils.Dictionaries import (
    team_index_07,
    team_index_08,
    team_index_12,
    team_index_13,
    team_index_14,
    team_index_current,
)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
ODDS_DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
TEAMS_DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"
OUTPUT_DB_PATH = BASE_DIR / "Data" / "dataset.sqlite"
OUTPUT_TABLE = "dataset_2012-27"

TEAM_INDEX_BY_SEASON = {
    "2007-08": team_index_07,
    "2008-09": team_index_08,
    "2009-10": team_index_08,
    "2010-11": team_index_08,
    "2011-12": team_index_08,
    "2012-13": team_index_12,
    "2013-14": team_index_13,
    "2014-15": team_index_14,
    "2015-16": team_index_14,
    "2016-17": team_index_14,
    "2017-18": team_index_14,
    "2018-19": team_index_14,
    "2019-20": team_index_14,
    "2020-21": team_index_14,
    "2021-22": team_index_14,
    "2022-23": team_index_current,
    "2023-24": team_index_current,
    "2024-25": team_index_current,
    "2025-26": team_index_current,
    "2026-27": team_index_current,
}


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def normalize_date(value):
    return calendar_date(value).isoformat()


def get_team_index_map(season_key):
    if season_key in TEAM_INDEX_BY_SEASON:
        return TEAM_INDEX_BY_SEASON[season_key]
    try:
        start_year = int(season_key.split("-")[0])
    except (ValueError, IndexError):
        return team_index_current
    return team_index_current if start_year >= 2022 else team_index_14


def fetch_team_table(teams_con, date_str, known_tables=None):
    if known_tables is None:
        if not table_exists(teams_con, date_str):
            return None
    elif date_str not in known_tables:
        return None
    return pd.read_sql_query(f'SELECT * FROM "{date_str}"', teams_con)


def build_game_features(team_df, home_team, away_team, index_map):
    home_index = index_map.get(home_team)
    away_index = index_map.get(away_team)
    if home_index is None or away_index is None:
        return None
    if len(team_df.index) != 30:
        return None

    home_team_series = team_df.iloc[home_index]
    away_team_series = team_df.iloc[away_index]
    return pd.concat([
        home_team_series,
        away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values}),
    ])


def select_odds_table(odds_con, season_key):
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        f"{season_key}",
    ]
    for table_name in candidates:
        if table_exists(odds_con, table_name):
            return table_name
    return None


def build_dataset(config=None, blend_k=0, blend_r=0.75, include_metadata=False):
    """Shared historical builder for persisted training data and walk-forward evaluation."""
    config = config or toml.load(CONFIG_PATH)
    validate_settings(blend_k, blend_r)
    metadata = []

    scores = []
    win_margin = []
    ou_values = []
    ou_cover = []
    games = []
    days_rest_away = []
    days_rest_home = []

    with sqlite3.connect(ODDS_DB_PATH) as odds_con, sqlite3.connect(TEAMS_DB_PATH) as teams_con:
        dates = snapshot_dates(teams_con)
        known_tables = set(dates)

        # Bound memory while reusing snapshots across every game on the same date.
        @lru_cache(maxsize=32)
        def cached_team_table(snapshot_date):
            return fetch_team_table(teams_con, snapshot_date, known_tables)

        @lru_cache(maxsize=None)
        def season_prior(season_key):
            return load_local_prior(teams_con, dates, season_key, config)

        @lru_cache(maxsize=32)
        def blended_table(snapshot_date, season_key):
            raw = cached_team_table(snapshot_date)
            return blend_team_stats(raw, season_prior(season_key), blend_k, blend_r) if blend_k else raw

        @lru_cache(maxsize=None)
        def prior_snapshot(game_date, season_start):
            return select_snapshot_date(dates, game_date, season_start)

        for season_key, season_config in config["create-games"].items():
            skipped_no_prior = 0
            season_start = normalize_date(datetime.strptime(season_config["start_date"], "%Y-%m-%d"))
            print(season_key)
            odds_table = select_odds_table(odds_con, season_key)
            if not odds_table:
                print(f"Missing odds tables for {season_key}. Skipped (no prior snapshot): 0")
                continue

            odds_df = pd.read_sql_query(f'SELECT * FROM "{odds_table}"', odds_con)
            if odds_df.empty:
                print(f"No odds data for {season_key}. Skipped (no prior snapshot): 0")
                continue

            index_map = get_team_index_map(season_key)

            for row in odds_df.itertuples(index=False):
                date_str = normalize_date(row.Date)
                snapshot_date = prior_snapshot(date_str, season_start)
                if snapshot_date is None:
                    skipped_no_prior += 1
                    continue
                team_df = blended_table(snapshot_date, season_key)

                game = build_game_features(team_df, row.Home, row.Away, index_map)
                if game is None:
                    continue

                # Keep game dates for chronological train/test splits, not snapshot dates.
                game["Date"] = date_str
                game["Date.1"] = date_str
                scores.append(row.Points)
                ou_values.append(row.OU)
                days_rest_home.append(row.Days_Rest_Home)
                days_rest_away.append(row.Days_Rest_Away)
                win_margin.append(1 if row.Win_Margin > 0 else 0)

                if row.Points < row.OU:
                    ou_cover.append(0)
                elif row.Points > row.OU:
                    ou_cover.append(1)
                else:
                    ou_cover.append(2)

                games.append(game)
                metadata.append({"Season": season_key, "Home_TEAM_ID": game["TEAM_ID"],
                                 "Away_TEAM_ID": game["TEAM_ID.1"], "ML_Home": getattr(row, "ML_Home", None),
                                 "ML_Away": getattr(row, "ML_Away", None)})

            print(f"Skipped (no prior snapshot) for {season_key}: {skipped_no_prior}")

    if not games:
        print("No game rows produced. Check odds and team tables.")
        return pd.DataFrame()

    season = pd.concat(games, ignore_index=True, axis=1).T
    frame = season.drop(columns=["TEAM_ID", "TEAM_ID.1"], errors="ignore")
    frame["Score"] = np.asarray(scores)
    frame["Home-Team-Win"] = np.asarray(win_margin)
    frame["OU"] = np.asarray(ou_values)
    frame["OU-Cover"] = np.asarray(ou_cover)
    frame["Days-Rest-Home"] = np.asarray(days_rest_home)
    frame["Days-Rest-Away"] = np.asarray(days_rest_away)

    for field in frame.columns.values:
        if "TEAM_" in field or "Date" in field:
            continue
        frame[field] = frame[field].astype(float)

    if include_metadata:
        for col in ("Season", "ML_Home", "ML_Away", "Home_TEAM_ID", "Away_TEAM_ID"):
            frame[col] = [row[col] for row in metadata]
    return frame


def blend_dataset(raw, k, r, priors):
    """Fast evaluation adapter: same team-frame math without rereading game rows.

    Raw metadata supplies stable team IDs. GP is left raw on both sides. A prior
    dictionary (including None for absent seasons) is shared across the sweep.
    """
    validate_settings(k, r)
    result = raw.copy()
    if not k:
        return result
    for season, rows in raw.groupby("Season", sort=False):
        for suffix, identity in [("", "Home_TEAM_ID"), (".1", "Away_TEAM_ID")]:
            cols = [name + suffix for name in TEAM_FEATURES]
            current = rows[cols].copy()
            current.columns = list(TEAM_FEATURES)
            current["TEAM_ID"] = rows[identity]
            blended = blend_team_stats(current, priors.get(season), k, r)
            result.loc[rows.index, cols] = blended[list(TEAM_FEATURES)].to_numpy()
    return result


def main(argv=None):
    config = toml.load(CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Build pre-game features with optional prior-season blending")
    parser.add_argument("--blend-k", type=float, default=config.get("blend_k", 0))
    parser.add_argument("--blend-r", type=float, default=config.get("blend_r", 0.75))
    args = parser.parse_args(argv)
    table = dataset_table(args.blend_k) if args.blend_k else OUTPUT_TABLE
    print(f"Blend k={args.blend_k:g}, r={args.blend_r:g}; output={table}")
    frame = build_dataset(config, args.blend_k, args.blend_r)
    if not frame.empty:
        with sqlite3.connect(OUTPUT_DB_PATH) as con:
            frame.to_sql(table, con, if_exists="replace", index=False)


if __name__ == "__main__":
    main()

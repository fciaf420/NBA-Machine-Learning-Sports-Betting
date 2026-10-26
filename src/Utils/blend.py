"""Dependency-free shrinkage math, with a pandas-compatible frame adapter.

Ranks are blended as numeric prior-strength signals (not re-ranked). This keeps
NBA's rank direction/tie convention and exactly the existing feature contract.
"""
from datetime import datetime
import math
import warnings

RAW_COLUMNS = {"TEAM_ID", "TEAM_NAME", "GP", "W", "L", "index", "level_0"}


def validate_settings(k, r):
    if not math.isfinite(k) or k < 0 or not math.isfinite(r) or not 0 <= r <= 1:
        raise ValueError("blend k must be finite and >= 0; r must be in [0, 1]")


def previous_season(season):
    year = int(season.split("-")[0]) - 1
    return f"{year}-{(year + 1) % 100:02d}"


def prior_snapshot_date(dates, season, config):
    """Inclusive configured previous-season bounds; never cross into this season.

    TeamData is collected with SeasonType=Regular Season even when the snapshot
    date falls in the playoffs, so the last snapshot contains final regular stats.
    """
    bounds = config["get-data"].get(previous_season(season))
    if not bounds:
        return None
    start = datetime.strptime(bounds["start_date"], "%Y-%m-%d").date().isoformat()
    end = datetime.strptime(bounds["end_date"], "%Y-%m-%d").date().isoformat()
    current_start = config["get-data"].get(season, {}).get("start_date")
    current_start = (datetime.strptime(current_start, "%Y-%m-%d").date().isoformat()
                     if current_start else f"{season[:4]}-09-01")
    eligible = [d for d in dates if start <= d <= end and d < current_start]
    return max(eligible, default=None)


PRIOR_POLICY = "local-teamdata-30teams-min58gp-v1"


def prior_is_complete(records):
    """58+ games covers shortened 2011/2020 seasons without accepting early data."""
    return (len(records) == 30
            and len({_team_id(row.get("TEAM_ID")) for row in records}) == 30
            and all(_number(row.get("TEAM_ID")) is not None
                    and (gp := _number(row.get("GP"))) is not None and gp >= 58
                    for row in records))


def load_local_prior(connection, dates, season, config, fill_missing=None):
    """Canonical training/serving source. API may fill an ABSENT local season.

    Existing incomplete history is never replaced by an API response. Both paths
    disable blending with the same completeness check. API fills are persisted,
    then reread through SQLite before use, so future dataset builds see the same
    exact prior. Caller caches this once per season.
    """
    selected = prior_snapshot_date(dates, season, config)
    if selected is None and fill_missing is not None:
        try:
            fetched = fill_missing(previous_season(season))
            records = fetched.to_dict("records")
            if not prior_is_complete(records):
                raise ValueError("need 30 unique teams with GP >= 58")
            bounds = config["get-data"][previous_season(season)]
            selected = datetime.strptime(bounds["end_date"], "%Y-%m-%d").date().isoformat()
            # Reapply season bounds even for a synthetic final snapshot date.
            if prior_snapshot_date([selected], season, config) != selected:
                raise ValueError("prior season date is not before current season")
            fetched.to_sql(selected, connection, if_exists="fail", index=False)
            connection.commit()
        except Exception as exc:
            warnings.warn(f"WARNING: cannot fill canonical prior for {season}: {exc}; effective blend_k=0")
            selected = None
    if selected is None:
        warnings.warn(f"WARNING: no complete prior-season TeamData for {season}; effective blend_k=0")
        return None
    cursor = connection.execute(f'SELECT * FROM "{selected}"')
    columns = [col[0] for col in cursor.description]
    records = [dict(zip(columns, row)) for row in cursor.fetchall()]
    if not prior_is_complete(records):
        warnings.warn(f"WARNING: incomplete prior {selected} for {season}: need 30 unique teams "
                      "with GP >= 58; effective blend_k=0")
        return None
    import pandas as pd
    frame = pd.DataFrame.from_records(records)
    frame.attrs["prior_date"] = selected
    frame.attrs["prior_policy"] = PRIOR_POLICY
    return frame


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _team_id(value):
    """SQLite/pandas may represent the same NBA ID as int, float, or text."""
    number = _number(value)
    return str(int(number)) if number is not None and number.is_integer() else str(value)


def blend_records(current, prior, k=0, r=0.75):
    """Pure-Python core; input/output lists of dictionaries, matched on TEAM_ID."""
    validate_settings(k, r)
    result = [dict(row) for row in current]
    if k == 0 or not prior:
        return result
    by_id = {_team_id(row["TEAM_ID"]): row for row in prior}
    if len(by_id) != len(prior):
        raise ValueError("Duplicate prior TEAM_ID")
    means = {}
    for col in current[0] if current else ():
        if col in RAW_COLUMNS:
            continue
        values = [v for row in prior if (v := _number(row.get(col))) is not None]
        if values:
            means[col] = sum(values) / len(values)
    for row in result:
        n = _number(row.get("GP"))
        if n is None or n < 0:
            raise ValueError("Current GP must be finite and nonnegative")
        weight = n / (n + k)
        old = by_id.get(_team_id(row["TEAM_ID"]), {})
        for col, mean in means.items():
            value = _number(row.get(col))
            if value is None:
                continue  # Preserve metadata and unavailable current statistics.
            old_value = _number(old.get(col))
            adjusted = r * (old_value if old_value is not None else mean) + (1 - r) * mean
            row[col] = weight * value + (1 - weight) * adjusted
    return result


def blend_team_stats(current, prior, k=0, r=0.75):
    """Return same columns/index/order; k=0 preserves values AND dtypes exactly."""
    validate_settings(k, r)
    if k == 0 or prior is None or prior.empty or current.empty:
        return current.copy()
    records = blend_records(current.to_dict("records"), prior.to_dict("records"), k, r)
    result = current.copy()
    for col in current.columns:
        if col not in RAW_COLUMNS:
            result[col] = [row[col] for row in records]
    return result


def dataset_table(k):
    validate_settings(k, 0.75)
    return "dataset_2012-27" + (f"_k{k:g}" if k else "")

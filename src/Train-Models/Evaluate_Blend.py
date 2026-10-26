"""Season walk-forward comparison. Does not persist datasets or trained models."""
import argparse
import importlib.util
import math
import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.Utils.features import XGB_ML_FEATURES, select_features
from src.Utils.gpu import xgb_device


def market_probability(home, away):
    """American moneylines -> no-vig home probability; invalid odds excluded."""
    def implied(value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return None
        if not math.isfinite(value) or abs(value) < 100:
            return None
        return 100 / (value + 100) if value > 0 else -value / (-value + 100)
    h, a = implied(home), implied(away)
    return None if h is None or a is None else h / (h + a)


def metrics(labels, probabilities):
    n = len(labels)
    if not n:
        return (float("nan"),) * 3
    p = [min(1 - 1e-7, max(1e-7, float(v))) for v in probabilities]
    loss = -sum(y * math.log(v) + (1 - y) * math.log(1 - v)
                for y, v in zip(labels, p)) / n
    brier = sum((v - y) ** 2 for y, v in zip(labels, p)) / n
    accuracy = sum((v >= 0.5) == bool(y) for y, v in zip(labels, p)) / n
    return loss, brier, accuracy


def main():
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", nargs="+", type=float, default=[0, 5, 10, 15, 20, 30])
    parser.add_argument("--r", nargs="+", type=float, default=[0.75, 1.0])
    parser.add_argument("--rounds", type=int, default=600)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("create_games", ROOT / "src/Process-Data/Create_Games.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    # Build/join OddsData only once; each sweep transforms the same raw games.
    raw = builder.build_dataset(include_metadata=True)
    if raw.empty:
        raise ValueError("No historical game data")
    config = builder.toml.load(builder.CONFIG_PATH)
    from src.Utils.blend import load_local_prior
    from src.Utils.snapshots import snapshot_dates
    with sqlite3.connect(builder.TEAMS_DB_PATH) as con:
        dates = snapshot_dates(con)
        priors = {season: load_local_prior(con, dates, season, config)
                  for season in raw.Season.unique()}
    records = []
    # k=0 is independent of r: compute/train it only once, copy reports for other r.
    zero_report = None
    for k in args.k:
        for r in args.r:
            if k == 0 and zero_report is not None:
                records.extend(dict(row, r=r) for row in zero_report)
                continue
            print(f"Building k={k:g}, r={r:g}", flush=True)
            frame = builder.blend_dataset(raw, k, r, priors)
            if frame.empty:
                raise ValueError("No historical game data")
            frame = frame.sort_values(["Season", "Date"], kind="stable").reset_index(drop=True)
            x = select_features(frame, XGB_ML_FEATURES).to_numpy(dtype=np.float32)
            y = frame["Home-Team-Win"].to_numpy(dtype=np.float32)
            all_predictions = []
            params = dict(objective="binary:logistic", eval_metric="logloss", max_depth=4,
                          eta=0.05, subsample=0.9, colsample_bytree=0.9,
                          tree_method="hist", device=xgb_device(), seed=42, nthread=4)
            for year in range(2021, 2026):
                season = f"{year}-{(year + 1) % 100:02d}"
                training_seasons = sorted(frame.loc[frame.Season < season, "Season"].unique())
                # Hold out whole most-recent seasons, not randomly selected games.
                n_valid = max(1, math.ceil(0.15 * len(training_seasons)))
                fit_seasons, valid_seasons = training_seasons[:-n_valid], training_seasons[-n_valid:]
                fit = frame.Season.isin(fit_seasons).to_numpy()
                valid = frame.Season.isin(valid_seasons).to_numpy()
                test = (frame.Season == season).to_numpy()
                if not fit.any() or not valid.any() or not test.any():
                    raise ValueError(f"Missing train/validation/test data for {season}")
                dfit, dvalid = xgb.DMatrix(x[fit], label=y[fit]), xgb.DMatrix(x[valid], label=y[valid])
                model = xgb.train(params, dfit, args.rounds, [(dvalid, "validation")],
                                  early_stopping_rounds=40, verbose_eval=False)
                rounds = model.best_iteration + 1
                # Once rounds are selected, refit on ALL prior seasons, never test.
                prior = fit | valid
                model = xgb.train(params, xgb.DMatrix(x[prior], label=y[prior]), rounds)
                prediction = model.predict(xgb.DMatrix(x[test]))
                for i, prob in zip(np.flatnonzero(test), prediction):
                    row = frame.iloc[i]
                    market = market_probability(row.ML_Home, row.ML_Away)
                    # Both sources scored on the exact same market-covered games.
                    all_predictions.append((season, float(y[i]), float(prob), market, float(row.GP)))
                print(f"  {season}: {test.sum()} games, {rounds} rounds", flush=True)
            result = pd.DataFrame(all_predictions, columns=["season", "y", "model", "market", "gp"])
            print(f"Market-covered: {result.market.notna().sum()}/{len(result)} test games")
            result = result[result.market.notna()]
            if result.empty:
                raise ValueError("No valid ML_Home/ML_Away pairs for market comparison")
            start = len(records)
            for season in ["ALL"] + sorted(result.season.unique()):
                sample = result if season == "ALL" else result[result.season == season]
                for bucket, mask in [("all", sample.gp >= 0), ("1-5", sample.gp.between(1, 5)),
                                     ("6-15", sample.gp.between(6, 15)), ("16+", sample.gp >= 16)]:
                    subset = sample[mask]
                    for source in ("model", "market"):
                        loss, brier, accuracy = metrics(subset.y.tolist(), subset[source].tolist())
                        records.append(dict(k=k, r=r, season=season, bucket=bucket, source=source,
                                            n=len(subset), log_loss=loss, brier=brier, accuracy=accuracy))
            if k == 0:
                zero_report = records[start:]
            output = pd.DataFrame(records)
            output.to_csv(ROOT / "Data/blend_eval.csv", index=False)
            print(output[(output.season == "ALL") & (output.k == k) & (output.r == r)]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    # Also save copied k=0 reports if that was the only requested strength.
    pd.DataFrame(records).to_csv(ROOT / "Data/blend_eval.csv", index=False)


if __name__ == "__main__":
    main()

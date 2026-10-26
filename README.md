# NBA Sports Betting Using Machine Learning
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

## Overview
This project predicts NBA game winners and totals (over/under) using team stats and sportsbook odds. It pulls team data from 2007-08 through the current season, builds matchup features, and runs trained models to estimate win probabilities and totals outcomes. It also outputs expected value and optional Kelly Criterion stake sizing.

## Features
- Moneyline and totals predictions (XGBoost and Neural Net models).
- Expected value calculation and optional Kelly Criterion sizing.
- Odds ingest from supported sportsbooks or manual input.
- Data processing pipeline and model training scripts.
- Flask web app for browsing outputs.

## How it works
1. **Collect stats and odds**: `Get_Data` pulls daily team stats from NBA endpoints and stores them in SQLite. `Get_Odds_Data` pulls sportsbook odds and scores from SBR and stores them in a separate SQLite DB.
2. **Build game features**: `Create_Games` merges team stats, odds, scores, and days-rest into a training dataset.
3. **Train models**: XGBoost/NN scripts in `src/Train-Models` fit moneyline and totals models.
4. **Predict today**: `main.py` fetches today’s schedule, builds matchup features, loads trained models, and prints predictions, expected value, and optional Kelly Criterion sizing.

## Requirements
- Python 3.11
- Packages: Tensorflow, XGBoost, NumPy, Pandas, Colorama, Tqdm, Requests, Scikit-learn

Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Quick start
```bash
python3 main.py -xgb -odds=fanduel
```

Odds will be fetched automatically when `-odds` is provided. Supported books:
`fanduel`, `draftkings`, `betmgm`, `pointsbet`, `caesars`, `wynn`, `bet_rivers_ny`

If `-odds` is omitted, the script will prompt for manual odds and totals.

Optional flags:
- `-nn` run neural network model
- `-xgb` run XGBoost model
- `-A` run all models
- `-kc` show Kelly Criterion bankroll fraction

## Prediction markets (Kalshi + Polymarket US)
```bash
python3 main.py -A -odds=kalshi        # moneyline odds from Kalshi, totals from Kalshi's ladder
python3 main.py -A -odds=polymarket    # moneyline odds from Polymarket US
python3 main.py -A -markets            # model win % vs both venues, after taker fees
python3 main.py -A -odds=kalshi -date 2026-10-28
```
Contract prices are converted to American odds after taker fees (Kalshi 0.07·p·(1−p), Polymarket US 0.0695·p·(1−p)), so EV and Kelly output use the real all-in cost. Both APIs are public; no keys needed.

The season, stats URL, and schedule file are derived from today's date (schedule auto-downloads from fixturedownload.com). XGBoost uses CUDA when a GPU is present. The NN loader falls back to a Keras 2.14 weight loader on newer TF/Keras (e.g. Colab).

## Flask web app
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

```bash
cd Flask
flask --debug run
```

## Data pipeline and training
```bash
# Create/update datasets
cd src/Process-Data
python -m Get_Data
python -m Get_Odds_Data
python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML --dataset dataset_2012-27 --trials 100 --splits 5 --calibration sigmoid
python -m XGBoost_Model_UO --dataset dataset_2012-27 --trials 100 --splits 5 --calibration sigmoid
python -m NN_Model_ML --dataset dataset_2012-27
python -m NN_Model_UO --dataset dataset_2012-27
python -m Logistic_Regression_ML --dataset dataset_2012-27 --trials 50 --splits 5 --calibration sigmoid
python -m Logistic_Regression_UO --dataset dataset_2012-27 --trials 50 --splits 5 --calibration sigmoid
```

### Neural network notes
- NN scripts accept `--dataset`, trial-count, seed, and chronological split flags.
- They train on `dataset_2012-27` and save into `Models/` with metric/parameter-based names.
- Rebuild the leak-free `dataset_2012-27` with Create_Games before retraining. NN training and serving prefer `tf_keras` when installed, otherwise native `tf.keras`; both use the standard Adam optimizer (no Keras-2-only legacy optimizer).

### Backfilling missing data
Get_Data normally fetches only new dates in the current season. To fill missing dates:
```bash
cd src/Process-Data
python -m Get_Data --backfill
```

To backfill a single season:
```bash
cd src/Process-Data
python -m Get_Data --backfill --season 2025-26
```

### Backfilling odds data
Get_Odds_Data normally fetches only new dates in the current season. To fill missing odds dates:
```bash
cd src/Process-Data
python -m Get_Odds_Data --backfill
```

To backfill a single season:
```bash
cd src/Process-Data
python -m Get_Odds_Data --backfill --season 2025-26
```

## Contributing
Contributions are welcome. If you change model behavior or data pipelines, add a note in the README and update any related scripts or docs.

### Optional prior-season blending

`config.toml` defaults to `blend_k = 0` (unchanged features) and `blend_r = 0.75`.
With strength `k`, stats use `GP/(GP+k)` current season and `k/(GP+k)` previous
season, regressed toward that prior league mean by `r`. IDs, names, GP, W and L
stay raw; W_PCT and numeric rank signals are blended. Feature names/order stay
unchanged. Prior snapshots are bounded to the configured **previous** season;
TeamData uses regular-season stats even for playoff-date snapshots. Missing prior
history warns and leaves that season unblended; missing individual teams use the
prior league mean. Current-season snapshots remain strictly before game day.

```bash
cd src/Train-Models
python -m Evaluate_Blend                     # optional --k 0 10 20 --r 0.75 1
# Data/blend_eval.csv: walk-forward 2021-22..2025-26, all + GP buckets,
# per-season + pooled scores; model and no-vig market use identical covered games.
cd ../Process-Data
PYTHONPATH=../.. python -m Create_Games --blend-k 10 --blend-r 0.75
cd ../Train-Models
python -m XGBoost_Model_ML --dataset dataset_2012-27_k10
python -m XGBoost_Model_UO --dataset dataset_2012-27_k10
python -m NN_Model_ML --dataset dataset_2012-27_k10
python -m NN_Model_UO --dataset dataset_2012-27_k10
```

Evaluation builds raw games once, sweeps shared blend math, selects boosting rounds
using the last 15% of prior **whole seasons**, then refits on all prior seasons.
It targets a Colab T4 run under roughly 20 minutes (runtime not yet benchmarked).
No trained models are saved by evaluation. GP buckets refer to pre-game snapshot GP.
After selecting settings, set **both** `blend_k` and `blend_r` in config before live
inference, matching the active trained models (`main.py -blend-k 10` overrides k).
Local TeamData is the canonical prior for both training and inference.
Both require 30 unique teams and GP >= 58 for every team. Incomplete priors warn
and disable blending for that season; inference then selects a k=0 model.
Only when no local prior exists may inference fetch final regular-season stats
from the API, validate them, persist them locally, then reread the same snapshot. Dataset table names encode k, **not r**: different r at the same k replaces
the table. Do not mix settings or compare already-selected hyperparameters as an
unbiased out-of-sample performance claim.


Models now carry `<model>.meta.json` provenance: dataset, k/r, feature schema and
order/count, prior snapshot dates, effective per-season k and creation time.
Dataset provenance is persisted in `dataset.sqlite` by Create_Games, not guessed
from the current config during training. Rebuild older blended tables before
training; blended tables without provenance are rejected. Runners select the
newest compatible model, never the highest historical accuracy. Missing metadata
is accepted only for legacy k=0 models; invalid metadata is rejected. If a model
includes the serving season in its prior-date history, that date must match too.
No compatible model means a clear error, not silently using mismatched weights.

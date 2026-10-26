"""Dependency-free dataset provenance and fail-closed model selection."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings

from src.Utils.blend import PRIOR_POLICY, validate_settings


def save_dataset_metadata(connection, dataset, metadata):
    connection.execute('CREATE TABLE IF NOT EXISTS _blend_metadata (dataset TEXT PRIMARY KEY, metadata TEXT NOT NULL)')
    connection.execute('INSERT OR REPLACE INTO _blend_metadata VALUES (?, ?)',
                       (dataset, json.dumps(metadata, allow_nan=False)))
    connection.commit()


def dataset_metadata(database, dataset):
    import sqlite3
    with closing(sqlite3.connect(database)) as con:
        exists = con.execute("SELECT 1 FROM sqlite_master WHERE name='_blend_metadata'").fetchone()
        row = con.execute('SELECT metadata FROM _blend_metadata WHERE dataset=?', (dataset,)).fetchone() if exists else None
    if row:
        metadata = json.loads(row[0])
        validate_settings(metadata['blend_k'], metadata['blend_r'])
        if metadata.get('prior_policy') != PRIOR_POLICY:
            raise ValueError("Dataset uses obsolete prior policy; rebuild Create_Games before training")
        return metadata
    if '_k' in dataset:
        raise ValueError(f"Missing blend provenance for {dataset}; rebuild with Create_Games before training")
    warnings.warn(f"Legacy dataset {dataset}: treating as blend_k=0")
    return dict(blend_k=0, blend_r=0.75, prior_dates={}, prior_policy=PRIOR_POLICY)


def unique_model_path(path):
    path = Path(path)
    return path.with_name(f'{path.stem}_{time.time_ns()}{path.suffix}')


def write_model_metadata(path, provenance, dataset, schema, columns):
    metadata = dict(provenance, dataset=dataset, feature_schema=schema,
                    feature_count=len(columns), features=list(columns),
                    created_at=datetime.now(timezone.utc).isoformat())
    Path(str(path) + '.meta.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata


def select_matching_model(candidates, k, r, schema, columns, season=None, prior_date=None):
    validate_settings(k, r)
    matches = []
    for path in candidates:
        path = Path(path)
        if path.name.endswith('.meta.json'):
            continue
        sidecar = Path(str(path) + '.meta.json')
        if not sidecar.exists():
            if k == 0:
                matches.append((path.stat().st_mtime, path))
            continue
        try:
            metadata = json.loads(sidecar.read_text())
            validate_settings(metadata['blend_k'], metadata['blend_r'])
            if (metadata['blend_k'] != k or (k != 0 and metadata['blend_r'] != r)
                    or metadata.get('feature_schema') != schema
                    or metadata.get('feature_count') != len(columns)
                    or metadata.get('features') != list(columns)
                    or metadata.get('prior_policy') != PRIOR_POLICY):
                continue
            prior_dates = metadata.get("prior_dates", {})
            if k and season in prior_dates and prior_dates[season] != prior_date:
                continue
            created = datetime.fromisoformat(metadata['created_at'])
            if created.tzinfo is None:
                raise ValueError('created_at must include timezone')
            matches.append((created.timestamp(), path))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            warnings.warn(f"Ignoring invalid model metadata {sidecar.name}: {exc}")
    if not matches:
        raise FileNotFoundError(f"No {schema} model matches active blend_k={k:g}, blend_r={r:g} "
                                f"and {len(columns)} features. Rebuild dataset and retrain matching models.")
    return max(matches, key=lambda item: (item[0], str(item[1])))[1]

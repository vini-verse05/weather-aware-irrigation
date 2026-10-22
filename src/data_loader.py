"""Dataset discovery, inspection, validation and cleaning.

Run ``python -m src.data_loader`` to locate the dataset under ``data/``, print
its real columns, the suggested column-to-role mapping, the sampling interval
and missing values, and to confirm that the required columns are present.

The dataset schema is NOT hard-coded. Columns are mapped to logical roles
(timestamp, soil_moisture, temperature, ...) by name heuristics that can be
overridden in ``config.local.json``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config


class DataValidationError(Exception):
    """Raised when the dataset is missing or does not satisfy requirements."""


class ColumnMappingError(DataValidationError):
    """Raised when required columns cannot be identified."""


@dataclass
class LoadedData:
    """Cleaned time series plus everything needed to interpret it."""

    df: pd.DataFrame                    # indexed by timestamp, sorted, unique
    mapping: dict[str, str | None]      # role -> column name
    interval: pd.Timedelta              # inferred sampling interval
    report: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    candidates: dict[str, list[str]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------
_FILE_PENALTY_HINTS = ("event", "raw", "valve", "dictionary", "readme", "metadata", "sensor_measure", "preprocessed")
_FILE_BONUS_HINTS = ("series", "10min", "grid", "sector", "merged")


def read_header(path: Path) -> list[str]:
    """Return the column names of a CSV file without reading the whole file."""
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # noqa: BLE001 - surface any parser problem uniformly
        raise DataValidationError(f"Could not read header of {path.name}: {exc}") from exc


def find_candidate_files(data_dir: Path | None = None) -> list[Path]:
    """Find CSV files under ``data_dir`` (recursively), best candidates first.

    A file is a candidate when its header contains a timestamp-like column and
    a soil-moisture-like column. Files whose names suggest raw exports or event
    tables are ranked lower.
    """
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    if not data_dir.exists():
        return []
    scored: list[tuple[int, Path]] = []
    patterns = (".csv", ".csv.gz")
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or not path.name.lower().endswith(patterns):
            continue
        try:
            cols = read_header(path)
        except DataValidationError:
            continue
        mapping, _ = detect_columns(cols)
        if mapping.get("timestamp") is None or mapping.get("soil_moisture") is None:
            continue
        name = path.name.lower()
        score = 0
        score += sum(2 for h in _FILE_BONUS_HINTS if h in name)
        score -= sum(3 for h in _FILE_PENALTY_HINTS if h in name)
        score += sum(1 for r in config.OPTIONAL_ROLES if mapping.get(r))
        scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [p for _, p in scored]


# --------------------------------------------------------------------------
# Column mapping
# --------------------------------------------------------------------------
def _is_excluded(name: str) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in config.EXCLUDED_NAME_PATTERNS)


def detect_columns(
    columns: list[str], overrides: dict[str, str] | None = None
) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    """Suggest a role -> column mapping from column names.

    Returns ``(mapping, candidates)``. ``candidates[role]`` lists every column
    that matched the role (the first is chosen unless overridden). Columns that
    look like engineered features/targets are never auto-selected.
    """
    overrides = overrides or {}
    mapping: dict[str, str | None] = {}
    candidates: dict[str, list[str]] = {}
    used: set[str] = set()
    all_roles = list(config.REQUIRED_ROLES) + list(config.OPTIONAL_ROLES)
    for role in all_roles:
        if role in overrides and overrides[role]:
            col = overrides[role]
            if col not in columns:
                raise ColumnMappingError(
                    f"column_overrides['{role}'] = '{col}' is not a column of the file. "
                    f"Available columns: {columns}"
                )
            mapping[role] = col
            candidates[role] = [col]
            used.add(col)
            continue
        found: list[str] = []
        for pattern in config.ROLE_PATTERNS[role]:
            for col in columns:
                if col in used or col in found or _is_excluded(col):
                    continue
                if re.search(pattern, col.lower()):
                    found.append(col)
        candidates[role] = found
        mapping[role] = found[0] if found else None
        if mapping[role]:
            used.add(mapping[role])
    return mapping, candidates


def detect_target_like_columns(columns: list[str]) -> list[str]:
    """Columns that look like a pre-computed future target (e.g. 24 h ahead)."""
    out = []
    for col in columns:
        low = col.lower()
        if any(re.search(p, low) for p in config.TARGET_LIKE_PATTERNS) and re.search(r"moist|swc|vwc|sm", low):
            out.append(col)
    return out


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def infer_sampling_interval(index: pd.DatetimeIndex) -> tuple[pd.Timedelta, float]:
    """Median spacing of a sorted DatetimeIndex and the share of steps equal to it."""
    if len(index) < 3:
        raise DataValidationError("Need at least 3 timestamps to infer a sampling interval.")
    diffs = pd.Series(index[1:] - index[:-1])
    interval = diffs.median()
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        raise DataValidationError("Could not infer a positive sampling interval from the timestamps.")
    regular = float((diffs == interval).mean())
    return pd.Timedelta(interval), regular


def clean_timeseries(
    raw: pd.DataFrame, mapping: dict[str, str | None]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse timestamps, sort, merge duplicates and blank invalid values.

    Returns the cleaned frame (DatetimeIndex, sorted, unique) and a report.
    Impossible values (negative soil moisture/precipitation, humidity outside
    0-100) are set to NaN rather than dropped; they are imputed later using
    training statistics only.
    """
    ts_col = mapping.get("timestamp")
    sm_col = mapping.get("soil_moisture")
    if not ts_col or ts_col not in raw.columns:
        raise ColumnMappingError("No timestamp column identified.")
    if not sm_col or sm_col not in raw.columns:
        raise ColumnMappingError("No soil-moisture column identified.")

    report: dict[str, Any] = {"rows_in": int(len(raw))}
    keep = [c for c in dict.fromkeys(mapping.values()) if c]
    df = raw[keep].copy()

    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    bad_ts = int(ts.isna().sum())
    df[ts_col] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=[ts_col])
    report["invalid_timestamps_dropped"] = bad_ts

    for role, col in mapping.items():
        if role == "timestamp" or not col:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    invalid = {}
    for role in ("soil_moisture", "precipitation", "irrigation", "et0"):
        col = mapping.get(role)
        if col and role != "et0":
            mask = df[col] < 0
            invalid[col] = int(mask.sum())
            df.loc[mask, col] = np.nan
    hcol = mapping.get("humidity")
    if hcol:
        lo, hi = config.HUMIDITY_VALID_RANGE
        mask = (df[hcol] < lo) | (df[hcol] > hi)
        invalid[hcol] = int(mask.sum())
        df.loc[mask, hcol] = np.nan
    report["invalid_values_blanked"] = invalid

    df = df.sort_values(ts_col, kind="mergesort")
    dup = int(df[ts_col].duplicated().sum())
    report["duplicate_timestamps_merged"] = dup
    if dup:
        df = df.groupby(ts_col, sort=True).mean(numeric_only=True)
    else:
        df = df.set_index(ts_col)
    df.index.name = "timestamp"
    df = df[[c for c in keep if c != ts_col]]

    report["rows_out"] = int(len(df))
    report["missing_by_column"] = {c: int(df[c].isna().sum()) for c in df.columns}
    if df.empty or df[sm_col].notna().sum() == 0:
        raise DataValidationError("No valid soil-moisture observations remain after cleaning.")
    report["start"] = str(df.index.min())
    report["end"] = str(df.index.max())
    return df, report


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _resolve_data_file(data_file: str | Path | None, overrides: dict[str, Any]) -> Path:
    if data_file is None and overrides.get("data_file"):
        data_file = overrides["data_file"]
    if data_file is not None:
        p = Path(data_file)
        if not p.is_absolute():
            p = config.PROJECT_ROOT / p
        if not p.exists():
            raise DataValidationError(f"Data file not found: {data_file}")
        return p
    candidates = find_candidate_files()
    if not candidates:
        raise DataValidationError(
            "No usable dataset found in the data/ folder. Download the dataset and extract it "
            "into data/ (see data/README.md), then run `python -m src.data_loader` to check it."
        )
    return candidates[0]


def load_dataset(
    data_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> LoadedData:
    """Locate, read, map and clean the dataset.

    ``overrides`` defaults to the contents of ``config.local.json``.
    """
    if overrides is None:
        overrides = config.load_local_overrides()
    path = _resolve_data_file(data_file, overrides)

    try:
        raw = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        raise DataValidationError(f"Could not read {path.name}: {exc}") from exc

    sector_col = overrides.get("sector_column")
    if sector_col:
        if sector_col not in raw.columns:
            raise ColumnMappingError(f"sector_column '{sector_col}' is not in {path.name}.")
        values = sorted(raw[sector_col].dropna().astype(str).unique())
        wanted = overrides.get("sector_value")
        if wanted is None:
            raise DataValidationError(
                f"{path.name} contains several sectors in column '{sector_col}': {values}. "
                "Set 'sector_value' in config.local.json to choose one."
            )
        raw = raw[raw[sector_col].astype(str) == str(wanted)]
        if raw.empty:
            raise DataValidationError(f"No rows for sector '{wanted}'. Available: {values}")

    mapping, candidates = detect_columns(list(raw.columns), overrides.get("column_overrides"))
    missing = [r for r in config.REQUIRED_ROLES if not mapping.get(r)]
    if missing:
        raise ColumnMappingError(
            f"Could not identify required column(s) {missing} in {path.name}. Columns found: "
            f"{list(raw.columns)}. Set 'column_overrides' in config.local.json."
        )

    df, report = clean_timeseries(raw, mapping)
    interval, regular = infer_sampling_interval(df.index)
    report["sampling_interval"] = str(interval)
    report["share_of_steps_at_interval"] = round(regular, 4)
    report["target_like_columns"] = detect_target_like_columns(list(raw.columns))
    report["all_columns"] = list(raw.columns)
    report["file"] = path.name
    if sector_col:
        report["sector"] = str(overrides.get("sector_value"))
    return LoadedData(df=df, mapping=mapping, interval=interval, report=report, path=path, candidates=candidates)


def validate_existing_target(
    raw: pd.DataFrame,
    mapping: dict[str, str | None],
    target_col: str,
    horizon_hours: int = config.FORECAST_HORIZON_HOURS,
    rel_tol: float = 0.02,
) -> dict[str, Any]:
    """Check a dataset-supplied future-target column against a timestamp-shifted one.

    Returns the share of rows where the supplied target agrees (within
    ``rel_tol`` of the soil-moisture range) with the value observed
    ``horizon_hours`` later. Used for validation only, never as a feature.
    """
    ts_col, sm_col = mapping["timestamp"], mapping["soil_moisture"]
    tmp = raw[[ts_col, sm_col, target_col]].copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce", utc=True).dt.tz_localize(None)
    tmp = tmp.dropna(subset=[ts_col]).drop_duplicates(ts_col).set_index(ts_col).sort_index()
    interval, _ = infer_sampling_interval(tmp.index)
    shifted = tmp[sm_col].reindex(tmp.index + pd.Timedelta(hours=horizon_hours), method="nearest", tolerance=interval / 2)
    shifted = pd.Series(shifted.to_numpy(), index=tmp.index)
    both = pd.concat([shifted, tmp[target_col]], axis=1).dropna()
    if both.empty:
        return {"comparable_rows": 0, "agreement": None}
    span = float(tmp[sm_col].max() - tmp[sm_col].min()) or 1.0
    close = (both.iloc[:, 0] - both.iloc[:, 1]).abs() <= rel_tol * span
    return {"comparable_rows": int(len(both)), "agreement": round(float(close.mean()), 4)}


# --------------------------------------------------------------------------
# CLI: inspection / validation
# --------------------------------------------------------------------------
def inspect(data_file: str | None = None) -> int:
    """Print a human-readable inspection report. Returns a process exit code."""
    try:
        files = find_candidate_files()
        print("Candidate files under data/ (best first):")
        if files:
            for f in files:
                print(f"  - {f.relative_to(config.PROJECT_ROOT)}")
        else:
            print("  (none)")
        data = load_dataset(data_file)
    except DataValidationError as exc:
        print(f"\nDATASET NOT READY: {exc}")
        return 2

    r = data.report
    print(f"\nFile used           : {data.path}")
    print(f"Rows in / out       : {r['rows_in']} / {r['rows_out']}")
    print(f"Time range          : {r['start']}  ->  {r['end']}")
    print(f"Sampling interval   : {r['sampling_interval']}  ({r['share_of_steps_at_interval']:.1%} of steps)")
    print(f"Duplicate timestamps merged : {r['duplicate_timestamps_merged']}")
    print(f"Invalid timestamps dropped  : {r['invalid_timestamps_dropped']}")
    print(f"Invalid values blanked      : {r['invalid_values_blanked']}")
    print("\nDetected column mapping (VERIFY against the dataset's data dictionary):")
    for role, col in data.mapping.items():
        alt = [c for c in data.candidates.get(role, []) if c != col]
        extra = f"   other candidates: {alt}" if alt else ""
        print(f"  {role:<14}: {col}{extra}")
    print("\nMissing values per mapped column:")
    for col, n in r["missing_by_column"].items():
        print(f"  {col:<30} {n:>8}  ({n / max(r['rows_out'], 1):.1%})")
    if r["target_like_columns"]:
        print(f"\nColumns that look like pre-computed future targets: {r['target_like_columns']}")
        print("They are NOT used as features. Training builds its own target from the timestamps.")
    print("\nAll columns in the file:")
    print("  " + ", ".join(r["all_columns"]))
    print("\nOK: required columns found. Next step: python -m src.train_model")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate the irrigation dataset.")
    parser.add_argument("--data-file", help="CSV file to inspect (default: auto-detect under data/)")
    args = parser.parse_args(argv)
    return inspect(args.data_file)


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # config validation and non-index utilities remain usable
    def tqdm(iterable, **_: object):
        return iterable

from .labels import PotassiumLabeler
from .utils import write_json


def _resolve_record_column(frame: pd.DataFrame) -> str:
    for candidate in ("path", "record_path", "filename", "record"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "record_list.csv must contain one of: path, record_path, filename, record"
    )


def _read_header_row(ecg_root: Path, row: dict) -> dict:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("wfdb is required to index MIMIC-IV-ECG") from exc

    relative = str(row["record_path"])
    relative_no_ext = relative[:-4] if relative.endswith(".hea") else relative
    absolute_no_ext = ecg_root / relative_no_ext
    header = wfdb.rdheader(str(absolute_no_ext))
    base_date = header.base_date
    base_time = header.base_time
    if base_date is None or base_time is None:
        raise ValueError("header has no base_date/base_time")
    ecg_time = datetime.combine(base_date, base_time)
    return {
        "subject_id": int(row["subject_id"]),
        "study_id": int(row["study_id"]),
        "record_path": relative_no_ext,
        "ecg_time": ecg_time.isoformat(sep=" "),
        "sampling_rate": float(header.fs),
        "signal_length": int(header.sig_len),
        "n_sig": int(header.n_sig),
        "lead_names": "|".join(header.sig_name),
        "index_error": "",
    }


def build_ecg_index(
    ecg_root: str | Path,
    output_csv: str | Path,
    workers: int = 16,
    limit: int | None = None,
) -> pd.DataFrame:
    """Index ECG header timestamps and signal metadata.

    MIMIC-IV-ECG's record list identifies the patient and waveform path, while
    the WFDB header supplies the shifted ECG acquisition date and time needed
    for matching to MIMIC-IV Clinical.
    """
    root = Path(ecg_root).expanduser().resolve()
    record_list_path = root / "record_list.csv"
    if not record_list_path.exists():
        raise FileNotFoundError(f"Missing {record_list_path}")
    records = pd.read_csv(record_list_path)
    record_column = _resolve_record_column(records)
    required = {"subject_id", "study_id"}
    if not required.issubset(records.columns):
        raise ValueError(f"record_list.csv is missing {required - set(records.columns)}")
    records = records.rename(columns={record_column: "record_path"})
    if limit is not None:
        records = records.head(limit)
    rows = records[["subject_id", "study_id", "record_path"]].to_dict("records")

    def safe_read(row: dict) -> dict:
        try:
            return _read_header_row(root, row)
        except Exception as exc:  # keep an auditable list of unreadable records
            return {
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "record_path": str(row["record_path"]),
                "ecg_time": "",
                "sampling_rate": math.nan,
                "signal_length": -1,
                "n_sig": -1,
                "lead_names": "",
                "index_error": f"{type(exc).__name__}: {exc}",
            }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        indexed = list(
            tqdm(pool.map(safe_read, rows), total=len(rows), desc="Indexing WFDB headers")
        )
    frame = pd.DataFrame(indexed)
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return frame


def _duckdb_reader_sql(path: Path) -> str:
    safe_path = str(path).replace("'", "''")
    if path.suffix == ".parquet":
        return f"read_parquet('{safe_path}')"
    return (
        f"read_csv_auto('{safe_path}', header=true, sample_size=100000, "
        "ignore_errors=false, union_by_name=true)"
    )


def _find_table(root: Path, stem: str) -> Path:
    candidates = (
        root / "hosp" / f"{stem}.parquet",
        root / "hosp" / f"{stem}.csv.gz",
        root / "hosp" / f"{stem}.csv",
        root / f"{stem}.parquet",
        root / f"{stem}.csv.gz",
        root / f"{stem}.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem} under {root}")


def _selected_itemids(config: dict) -> list[int]:
    ids = [int(x) for x in config["data"]["potassium_itemids"]["serum"]]
    if config["data"].get("include_whole_blood", False):
        ids.extend(int(x) for x in config["data"]["potassium_itemids"]["whole_blood"])
    return sorted(set(ids))


def validate_potassium_items(clinical_root: str | Path, itemids: Iterable[int]) -> pd.DataFrame:
    """Return dictionary rows for the configured potassium item IDs."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required to query MIMIC-IV Clinical") from exc

    root = Path(clinical_root).expanduser().resolve()
    dictionary = _find_table(root, "d_labitems")
    ids = ",".join(str(int(x)) for x in sorted(set(itemids)))
    query = f"""
        SELECT itemid, label, fluid, category
        FROM {_duckdb_reader_sql(dictionary)}
        WHERE itemid IN ({ids})
        ORDER BY itemid
    """
    frame = duckdb.sql(query).df()
    missing = set(itemids) - set(frame["itemid"].astype(int).tolist())
    if missing:
        raise ValueError(f"Configured potassium item IDs absent from d_labitems: {missing}")
    non_potassium = frame[~frame["label"].str.contains("Potassium", case=False, na=False)]
    if not non_potassium.empty:
        raise ValueError(f"Configured item IDs are not potassium items:\n{non_potassium}")
    return frame


def build_potassium_cohort(config: dict) -> tuple[pd.DataFrame, dict]:
    """Pair every ECG to the nearest eligible potassium lab within the time window."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required to build the MIMIC cohort") from exc

    data_cfg = config["data"]
    ecg_index_path = Path(data_cfg["ecg_index_csv"]).expanduser().resolve()
    clinical_root = Path(data_cfg["clinical_root"]).expanduser().resolve()
    output_path = Path(data_cfg["cohort_csv"]).expanduser().resolve()
    if not ecg_index_path.exists():
        raise FileNotFoundError(f"Build the ECG index first: {ecg_index_path}")

    labevents = _find_table(clinical_root, "labevents")
    itemids = _selected_itemids(config)
    item_dictionary = validate_potassium_items(clinical_root, itemids)
    window = int(data_cfg["lab_window_minutes"])
    min_k = float(data_cfg["min_potassium"])
    max_k = float(data_cfg["max_potassium"])
    item_sql = ",".join(str(x) for x in itemids)
    before_first = bool(data_cfg.get("prefer_lab_before_ecg", True))
    tie_break = (
        "CASE WHEN l.charttime <= e.ecg_time THEN 0 ELSE 1 END,"
        if before_first
        else ""
    )

    ecg_reader = _duckdb_reader_sql(ecg_index_path)
    lab_reader = _duckdb_reader_sql(labevents)
    query = f"""
        WITH ecgs AS (
            SELECT
                CAST(subject_id AS BIGINT) AS subject_id,
                CAST(study_id AS BIGINT) AS study_id,
                record_path,
                CAST(ecg_time AS TIMESTAMP) AS ecg_time,
                CAST(sampling_rate AS DOUBLE) AS sampling_rate,
                CAST(signal_length AS BIGINT) AS signal_length,
                CAST(n_sig AS INTEGER) AS n_sig,
                lead_names
            FROM {ecg_reader}
            WHERE COALESCE(index_error, '') = ''
              AND ecg_time IS NOT NULL
        ),
        labs AS (
            SELECT
                CAST(subject_id AS BIGINT) AS subject_id,
                CAST(labevent_id AS BIGINT) AS labevent_id,
                CAST(itemid AS INTEGER) AS itemid,
                CAST(charttime AS TIMESTAMP) AS charttime,
                CAST(valuenum AS DOUBLE) AS potassium,
                valueuom,
                flag,
                priority
            FROM {lab_reader}
            WHERE itemid IN ({item_sql})
              AND valuenum BETWEEN {min_k} AND {max_k}
              AND charttime IS NOT NULL
        ),
        candidates AS (
            SELECT
                e.*,
                l.labevent_id,
                l.itemid AS potassium_itemid,
                l.charttime AS potassium_time,
                l.potassium,
                l.valueuom AS potassium_unit,
                l.flag AS potassium_flag,
                l.priority AS potassium_priority,
                date_diff('second', e.ecg_time, l.charttime) / 60.0 AS delta_minutes
            FROM ecgs e
            INNER JOIN labs l USING (subject_id)
            WHERE l.charttime BETWEEN
                e.ecg_time - INTERVAL '{window} minutes'
                AND e.ecg_time + INTERVAL '{window} minutes'
        )
        SELECT *
        FROM candidates
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY study_id
            ORDER BY ABS(delta_minutes), {tie_break} labevent_id
        ) = 1
        ORDER BY subject_id, ecg_time, study_id
    """
    cohort = duckdb.sql(query).df()
    labeler = PotassiumLabeler.from_config(config)
    cohort["label_id"] = labeler.transform(cohort["potassium"].to_numpy())
    cohort["label"] = labeler.label_names(cohort["label_id"].to_numpy())

    if data_cfg.get("require_12_leads", True):
        cohort = cohort.loc[cohort["n_sig"] == 12].copy()

    if cohort["study_id"].duplicated().any():
        raise AssertionError("Each ECG must pair to at most one potassium result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_path, index=False)

    counts = cohort["label"].value_counts().reindex(labeler.names, fill_value=0)
    summary = {
        "mimic_ecg_version": data_cfg["mimic_ecg_version"],
        "mimic_clinical_version": data_cfg["mimic_clinical_version"],
        "lab_window_minutes": window,
        "potassium_itemids": itemids,
        "potassium_item_dictionary": item_dictionary.to_dict("records"),
        "records": int(len(cohort)),
        "subjects": int(cohort["subject_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_prevalence": {
            str(k): float(v / len(cohort)) if len(cohort) else 0.0 for k, v in counts.items()
        },
        "median_abs_time_delta_minutes": (
            float(cohort["delta_minutes"].abs().median()) if len(cohort) else None
        ),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return cohort, summary

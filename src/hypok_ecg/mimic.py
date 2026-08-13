from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath
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
    missing_signal_files = sorted(
        {
            str(file_name)
            for file_name in header.file_name
            if not (absolute_no_ext.parent / str(file_name)).is_file()
        }
    )
    if missing_signal_files:
        raise FileNotFoundError(
            f"header references missing waveform files: {missing_signal_files}"
        )
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


def _normalize_precomputed_path(value: object) -> str:
    path = str(value).strip().replace("\\", "/")
    if path.endswith(".hea") or path.endswith(".dat"):
        path = path[:-4]
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"Unsafe or invalid relative ECG path: {value!r}")
    return path


def normalize_precomputed_cohort(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Validate and normalize a provided ECG-potassium cohort.

    The source file is assumed to contain already matched potassium values. This
    function validates its schema and labels, but cannot independently verify the
    upstream laboratory matching without labevents and potassium timestamps.
    """
    aliases = {
        "path": "record_path",
        "potassium_value": "potassium",
        "k_label": "label",
    }
    normalized = frame.rename(
        columns={source: target for source, target in aliases.items() if target not in frame}
    ).copy()
    required = {"subject_id", "study_id", "ecg_time", "record_path", "potassium", "label"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"Precomputed cohort is missing columns: {sorted(missing)}")
    if normalized.empty:
        raise ValueError("Precomputed cohort is empty")

    for column in ("subject_id", "study_id"):
        values = pd.to_numeric(normalized[column], errors="raise")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain finite integer identifiers")
        normalized[column] = values.astype(np.int64)
    normalized["potassium"] = pd.to_numeric(normalized["potassium"], errors="raise")
    normalized["provided_ecg_time"] = pd.to_datetime(
        normalized["ecg_time"], errors="raise"
    )
    normalized = normalized.drop(columns=["ecg_time"])
    normalized["record_path"] = normalized["record_path"].map(
        _normalize_precomputed_path
    )
    normalized["label"] = normalized["label"].astype(str).str.strip()

    if normalized["study_id"].duplicated().any():
        duplicates = int(normalized["study_id"].duplicated().sum())
        raise ValueError(f"Precomputed cohort contains {duplicates} duplicate study_id values")
    path_studies = normalized["record_path"].map(lambda value: PurePosixPath(value).name)
    expected_paths = normalized["study_id"].astype(str)
    if not path_studies.eq(expected_paths).all():
        raise ValueError("Some ECG paths do not end with their study_id")

    data_cfg = config["data"]
    in_range = normalized["potassium"].between(
        float(data_cfg["min_potassium"]),
        float(data_cfg["max_potassium"]),
        inclusive="both",
    )
    if not in_range.all():
        raise ValueError(f"Precomputed cohort has {int((~in_range).sum())} out-of-range K values")

    labeler = PotassiumLabeler.from_config(config)
    normalized["label_id"] = labeler.transform(normalized["potassium"].to_numpy())
    expected_labels = pd.Series(
        labeler.label_names(normalized["label_id"].to_numpy()),
        index=normalized.index,
    )
    mismatch = ~normalized["label"].eq(expected_labels)
    if mismatch.any():
        raise ValueError(
            f"Precomputed cohort has {int(mismatch.sum())} labels inconsistent with K thresholds"
        )
    return normalized


def _complete_standard_leads(lead_names: object, required_leads: list[str]) -> bool:
    observed = str(lead_names).split("|")
    return len(observed) == len(required_leads) and set(observed) == set(required_leads)


def _signal_value_quality(signal: np.ndarray, lead_names: list[str]) -> dict:
    """Summarize non-finite physical samples before a record enters a split."""
    array = np.asarray(signal)
    if array.ndim != 2:
        raise ValueError(f"Expected samples x leads, got {array.shape}")
    if array.shape[1] != len(lead_names):
        raise ValueError(
            f"Signal has {array.shape[1]} columns but {len(lead_names)} lead names"
        )
    nonfinite = ~np.isfinite(array)
    nonfinite_count = int(nonfinite.sum())
    affected_leads = [
        str(lead_names[index])
        for index in np.flatnonzero(nonfinite.any(axis=0))
    ]
    return {
        "signal_samples": int(array.size),
        "nonfinite_samples": nonfinite_count,
        "nonfinite_fraction": (
            float(nonfinite_count / array.size) if array.size else 0.0
        ),
        "nonfinite_leads": "|".join(affected_leads),
    }


def _read_precomputed_row(
    ecg_root: Path,
    row: dict,
    validate_signal_values: bool,
) -> dict:
    result = _read_header_row(ecg_root, row)
    if not validate_signal_values:
        result.update(
            {
                "signal_samples": -1,
                "nonfinite_samples": -1,
                "nonfinite_fraction": math.nan,
                "nonfinite_leads": "",
            }
        )
        return result
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("wfdb is required to validate MIMIC-IV-ECG signals") from exc

    record = wfdb.rdrecord(str(ecg_root / result["record_path"]))
    if record.p_signal is None:
        raise ValueError("physical ECG signal is unavailable")
    result.update(_signal_value_quality(record.p_signal, list(record.sig_name)))
    return result


def build_precomputed_cohort(
    config: dict,
    workers: int = 16,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Prepare an externally matched cohort and validate its local ECG files."""
    data_cfg = config["data"]
    source_path = Path(data_cfg["precomputed_cohort_csv"]).expanduser().resolve()
    output_path = Path(data_cfg["cohort_csv"]).expanduser().resolve()
    ecg_root = Path(data_cfg["ecg_root"]).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Missing precomputed cohort: {source_path}")
    if not ecg_root.exists():
        raise FileNotFoundError(f"Missing ECG root: {ecg_root}")

    source = pd.read_csv(source_path)
    if limit is not None:
        source = source.head(limit)
    prepared = normalize_precomputed_cohort(source, config).reset_index(drop=True)
    prepared["_source_row"] = np.arange(len(prepared), dtype=np.int64)
    rows = prepared[["_source_row", "subject_id", "study_id", "record_path"]].to_dict(
        "records"
    )
    validate_signal_values = bool(
        data_cfg.get("precomputed_validate_signal_values", True)
    )

    def safe_read(row: dict) -> dict:
        try:
            result = _read_precomputed_row(
                ecg_root,
                row,
                validate_signal_values=validate_signal_values,
            )
            result["_source_row"] = int(row["_source_row"])
            return result
        except Exception as exc:
            return {
                "_source_row": int(row["_source_row"]),
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "record_path": str(row["record_path"]),
                "ecg_time": "",
                "sampling_rate": math.nan,
                "signal_length": -1,
                "n_sig": -1,
                "lead_names": "",
                "index_error": f"{type(exc).__name__}: {exc}",
                "signal_samples": -1,
                "nonfinite_samples": -1,
                "nonfinite_fraction": math.nan,
                "nonfinite_leads": "",
            }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        indexed = list(
            tqdm(
                pool.map(safe_read, rows),
                total=len(rows),
                desc="Validating selected ECG files",
            )
        )
    metadata = pd.DataFrame(indexed)
    cohort = prepared.merge(
        metadata.drop(columns=["subject_id", "study_id", "record_path"]),
        on="_source_row",
        how="left",
        validate="one_to_one",
    )
    cohort["header_ecg_time"] = pd.to_datetime(cohort["ecg_time"], errors="coerce")
    cohort["ecg_time_delta_seconds"] = (
        cohort["header_ecg_time"] - cohort["provided_ecg_time"]
    ).dt.total_seconds()

    required_leads = list(data_cfg["lead_order"])
    cohort["complete_standard_leads"] = cohort["lead_names"].map(
        lambda value: _complete_standard_leads(value, required_leads)
    )
    readable = cohort["index_error"].fillna("").eq("")
    tolerance = float(data_cfg.get("precomputed_ecg_time_tolerance_seconds", 1.0))
    time_match = cohort["ecg_time_delta_seconds"].abs().le(tolerance)
    complete_leads = cohort["n_sig"].eq(len(required_leads)) & cohort[
        "complete_standard_leads"
    ]
    finite_signal = (
        cohort["nonfinite_samples"].eq(0)
        if validate_signal_values
        else pd.Series(True, index=cohort.index)
    )

    cohort["exclusion_reason"] = ""
    cohort.loc[~readable, "exclusion_reason"] = "unreadable_or_missing_waveform"
    cohort.loc[readable & ~time_match, "exclusion_reason"] = "ecg_time_mismatch"
    if data_cfg.get("require_12_leads", True):
        cohort.loc[readable & time_match & ~complete_leads, "exclusion_reason"] = (
            "incomplete_standard_leads"
        )
    lead_eligible = (
        complete_leads
        if data_cfg.get("require_12_leads", True)
        else pd.Series(True, index=cohort.index)
    )
    cohort.loc[
        readable & time_match & lead_eligible & ~finite_signal,
        "exclusion_reason",
    ] = "nonfinite_signal"
    keep = cohort["exclusion_reason"].eq("")

    audit_columns = [
        "subject_id",
        "study_id",
        "record_path",
        "provided_ecg_time",
        "header_ecg_time",
        "ecg_time_delta_seconds",
        "n_sig",
        "lead_names",
        "signal_samples",
        "nonfinite_samples",
        "nonfinite_fraction",
        "nonfinite_leads",
        "index_error",
        "exclusion_reason",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_path = output_path.with_name(f"{output_path.stem}.excluded.csv")
    cohort.loc[~keep, audit_columns].to_csv(excluded_path, index=False)

    cohort = cohort.loc[keep].copy()
    cohort["ecg_time"] = cohort["header_ecg_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cohort["cohort_source"] = "precomputed_unverified_matching"
    final_columns = [
        "subject_id",
        "study_id",
        "record_path",
        "ecg_time",
        "sampling_rate",
        "signal_length",
        "n_sig",
        "lead_names",
        "potassium",
        "label_id",
        "label",
        "cohort_source",
    ]
    cohort = cohort[final_columns].sort_values(
        ["subject_id", "ecg_time", "study_id"]
    )
    cohort.to_csv(output_path, index=False)

    labeler = PotassiumLabeler.from_config(config)
    counts = cohort["label"].value_counts().reindex(labeler.names, fill_value=0)
    summary = {
        "cohort_source": "precomputed",
        "matching_independently_verified": False,
        "matching_assumption": data_cfg.get("precomputed_matching_assumption", {}),
        "mimic_ecg_version": data_cfg["mimic_ecg_version"],
        "mimic_clinical_version": data_cfg.get("mimic_clinical_version", "stated 3.1"),
        "source_records": int(len(prepared)),
        "signal_values_validated": validate_signal_values,
        "header_errors": int((~readable).sum()),
        "ecg_time_mismatches": int((readable & ~time_match).sum()),
        "incomplete_standard_leads": int((readable & time_match & ~complete_leads).sum()),
        "nonfinite_waveforms": int(
            (readable & time_match & lead_eligible & ~finite_signal).sum()
        ),
        "records": int(len(cohort)),
        "subjects": int(cohort["subject_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_prevalence": {
            str(k): float(v / len(cohort)) if len(cohort) else 0.0 for k, v in counts.items()
        },
        "excluded_records_csv": str(excluded_path),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return cohort, summary


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


def build_cohort(
    config: dict,
    workers: int = 16,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the configured cohort from Clinical tables or a precomputed CSV."""
    source = str(config["data"].get("cohort_source", "clinical")).lower()
    if source == "precomputed":
        return build_precomputed_cohort(config, workers=workers, limit=limit)
    if source == "clinical":
        if limit is not None:
            raise ValueError("--limit is supported only for a precomputed cohort")
        return build_potassium_cohort(config)
    raise ValueError(f"Unsupported data.cohort_source: {source}")

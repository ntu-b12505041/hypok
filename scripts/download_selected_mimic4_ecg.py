#!/usr/bin/env python3
"""Selectively download MIMIC-IV-ECG WFDB records listed in a cohort CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *, total: int, desc: str) -> None:
            self.total = total
            self.desc = desc
            self.completed = 0

        def __enter__(self):
            print(f"{self.desc}: 0/{self.total}")
            return self

        def __exit__(self, *_: object) -> None:
            print(f"{self.desc}: {self.completed}/{self.total}")

        def update(self, amount: int) -> None:
            self.completed += amount
            if self.completed % 1000 == 0 or self.completed == self.total:
                print(f"{self.desc}: {self.completed}/{self.total}")


DEFAULT_BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0"


@dataclass(frozen=True)
class DownloadTask:
    url: str
    destination: Path
    record_path: str
    extension: str


def _safe_record_path(value: object) -> str:
    path = str(value).strip().replace("\\", "/")
    if path.endswith(".hea") or path.endswith(".dat"):
        path = path[:-4]
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"Unsafe or invalid ECG path: {value!r}")
    if parsed.parts[0] != "files":
        raise ValueError(f"ECG path must start with 'files/': {value!r}")
    return path


def read_record_paths(csv_path: Path, limit: int | None = None) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        path_column = next(
            (name for name in ("path", "record_path") if name in reader.fieldnames),
            None,
        )
        if path_column is None:
            raise ValueError("CSV must contain path or record_path")
        paths: list[str] = []
        seen: set[str] = set()
        for row in reader:
            path = _safe_record_path(row[path_column])
            if path not in seen:
                seen.add(path)
                paths.append(path)
            if limit is not None and len(paths) >= limit:
                break
    if not paths:
        raise ValueError("CSV contains no ECG paths")
    return paths


def make_tasks(
    record_paths: list[str],
    output_root: Path,
    base_url: str,
) -> list[DownloadTask]:
    root_url = base_url.rstrip("/")
    tasks: list[DownloadTask] = []
    for record_path in record_paths:
        encoded_path = "/".join(quote(part) for part in PurePosixPath(record_path).parts)
        for extension in ("hea", "dat"):
            tasks.append(
                DownloadTask(
                    url=f"{root_url}/{encoded_path}.{extension}",
                    destination=output_root / f"{record_path}.{extension}",
                    record_path=record_path,
                    extension=extension,
                )
            )
    return tasks


def _expected_total(response, existing_bytes: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/(\d+)$", content_range)
    if match:
        return int(match.group(1))
    content_length = response.headers.get("Content-Length")
    if content_length and response.status == 200:
        return int(content_length)
    if content_length and response.status == 206:
        return existing_bytes + int(content_length)
    return None


def _download_once(task: DownloadTask, timeout: float) -> str:
    destination = task.destination
    if destination.is_file() and destination.stat().st_size > 0:
        return "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    existing_bytes = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "hypok-ecg-selective-downloader/1.0"}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
    request = Request(task.url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code == 416 and partial.exists():
            partial.unlink()
        raise
    with response:
        append = existing_bytes > 0 and response.status == 206
        if existing_bytes > 0 and not append:
            existing_bytes = 0
        mode = "ab" if append else "wb"
        expected = _expected_total(response, existing_bytes)
        with partial.open(mode) as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
    actual = partial.stat().st_size
    if expected is not None and actual != expected:
        raise IOError(f"incomplete response: expected {expected} bytes, received {actual}")
    os.replace(partial, destination)
    return "downloaded"


def download_with_retries(
    task: DownloadTask,
    timeout: float,
    retries: int,
) -> tuple[str, DownloadTask, str]:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            return _download_once(task, timeout), task, ""
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    return "failed", task, last_error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download only MIMIC-IV-ECG .hea/.dat files listed in a CSV."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    csv_path = args.csv.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if args.workers < 1 or args.retries < 0:
        raise ValueError("workers must be >=1 and retries must be >=0")
    record_paths = read_record_paths(csv_path, limit=args.limit)
    tasks = make_tasks(record_paths, output_root, args.base_url)
    plan = {
        "records": len(record_paths),
        "files": len(tasks),
        "estimated_signal_gib": round(
            len(record_paths) * 5000 * 12 * 2 / (1024**3), 2
        ),
        "output_root": str(output_root),
        "base_url": args.base_url,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        first = tasks[:2]
        print(
            json.dumps(
                {
                    "first_urls": [task.url for task in first],
                    "first_destinations": [str(task.destination) for task in first],
                },
                indent=2,
            )
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with tqdm(total=len(tasks), desc="Downloading selected ECG files") as progress:
            batch_size = max(args.workers * 8, 64)
            for start in range(0, len(tasks), batch_size):
                futures = [
                    pool.submit(
                        download_with_retries,
                        task,
                        args.timeout,
                        args.retries,
                    )
                    for task in tasks[start : start + batch_size]
                ]
                for future in as_completed(futures):
                    status, task, error = future.result()
                    counts[status] += 1
                    if status == "failed":
                        failures.append((task.url, str(task.destination), error))
                    progress.update(1)

    failure_path = output_root / "download_failures.csv"
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["url", "destination", "error"])
        writer.writerows(failures)
    summary = {
        **plan,
        **counts,
        "failure_csv": str(failure_path),
    }
    summary_path = output_root / "selective_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

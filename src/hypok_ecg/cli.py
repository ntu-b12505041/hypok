from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import compare_runs
from .config import load_config
from .evaluation import evaluate_model
from .mimic import build_cohort, build_ecg_index
from .splits import write_splits
from .training import train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypok-ecg",
        description="MIMIC-IV ECG dyskalemia research pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def configured(name: str, help_text: str) -> argparse.ArgumentParser:
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/mimic.yaml")
        return command

    index = configured("index-ecg", "Index ECG timestamps from WFDB headers.")
    index.add_argument("--workers", type=int, default=16)
    index.add_argument("--limit", type=int)
    cohort = configured(
        "build-cohort",
        "Build from Clinical labs or validate an externally matched cohort.",
    )
    cohort.add_argument("--workers", type=int, default=16)
    cohort.add_argument("--limit", type=int)
    configured("split", "Create leakage-safe patient-level data splits.")
    configured("train", "Train and calibrate the multitask SE-ResNet.")
    configured("evaluate", "Evaluate once on the locked test split.")
    run_all = configured(
        "run-all", "Run cohort construction, split, training, and locked test evaluation."
    )
    run_all.add_argument("--workers", type=int, default=16)
    run_all.add_argument("--limit", type=int)
    configured("validate-config", "Validate configuration without reading data.")
    compare = subparsers.add_parser(
        "compare", help="Create a paired scratch-vs-pretrained model comparison."
    )
    compare.add_argument("--baseline-config", default="configs/mimic.yaml")
    compare.add_argument(
        "--finetune-config", default="configs/ecgfounder_finetune.yaml"
    )
    compare.add_argument("--output-dir", default="outputs/model_comparison")
    return parser


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare":
        report = compare_runs(
            args.baseline_config, args.finetune_config, args.output_dir
        )
        _print({"report": str(report)})
        return 0
    config = load_config(args.config)
    data = config["data"]

    if args.command == "validate-config":
        _print({"status": "ok", "config": str(Path(args.config).resolve())})
        return 0
    cohort_source = str(data.get("cohort_source", "clinical")).lower()
    should_build_full_index = args.command == "index-ecg" or (
        args.command == "run-all" and cohort_source == "clinical"
    )
    if should_build_full_index:
        frame = build_ecg_index(
            data["ecg_root"],
            data["ecg_index_csv"],
            workers=getattr(args, "workers", 16),
            limit=getattr(args, "limit", None),
        )
        _print(
            {
                "indexed": len(frame),
                "errors": int((frame["index_error"] != "").sum()),
                "path": data["ecg_index_csv"],
            }
        )
        if args.command == "index-ecg":
            return 0
    if args.command in {"build-cohort", "run-all"}:
        _, summary = build_cohort(
            config,
            workers=getattr(args, "workers", 16),
            limit=getattr(args, "limit", None),
        )
        _print(summary)
        if args.command == "build-cohort":
            return 0
    if args.command in {"split", "run-all"}:
        _, summary = write_splits(config)
        _print(summary)
        if args.command == "split":
            return 0
    if args.command in {"train", "run-all"}:
        _print(train_model(config))
        if args.command == "train":
            return 0
    if args.command in {"evaluate", "run-all"}:
        result = evaluate_model(config)
        _print(
            {
                "report": result["report"],
                "target_met": result["metrics"]["target"]["met"],
            }
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

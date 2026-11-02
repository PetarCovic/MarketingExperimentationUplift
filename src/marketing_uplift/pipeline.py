"""Download, validate, and benchmark the Criteo uplift dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import requests

from .experiment import analyze_experiment
from .models import fit_and_evaluate, FEATURES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "criteo-uplift-v2.1.csv.gz"
DEFAULT_FULL = ROOT / "data" / "processed" / "criteo_full.parquet"
DEFAULT_SAMPLE = ROOT / "data" / "processed" / "criteo_model_sample.parquet"
DEFAULT_OUTPUT = ROOT / "outputs" / "metrics.json"
DEFAULT_SMOKE_SOURCE = ROOT / "data" / "smoke" / "synthetic-smoke.csv.gz"
DATA_URL = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"


def _quote_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def relation_for(path: Path) -> str:
    quoted = _quote_path(path)
    return f"read_parquet('{quoted}')" if path.suffix == ".parquet" else f"read_csv_auto('{quoted}', header=true)"


def download(path: Path, accept_license: bool) -> dict[str, Any]:
    if not accept_license:
        raise ValueError("Pass --i-accept-license after reviewing the dataset's CC BY-NC-SA 4.0 license.")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    temp_path = path.with_name(path.name + ".download")
    try:
        with requests.get(DATA_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
        if byte_count == 0:
            raise ValueError("Download was empty")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    manifest = {"url": DATA_URL, "bytes": byte_count, "sha256": digest.hexdigest(), "license": "CC BY-NC-SA 4.0", "source": "Criteo Uplift Prediction Dataset v2.1", "data_kind": "official_criteo_v2.1"}
    (path.parent / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Overwrite any smoke-test sidecar attached to this same raw-data path.
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def generate_sample(path: Path, rows: int, seed: int = 42) -> dict[str, Any]:
    if rows < 100:
        raise ValueError("Synthetic rows must be at least 100")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 12))
    treatment = rng.binomial(1, 0.85, rows)
    base_logit = -2.5 + 0.4 * features[:, 0] - 0.25 * features[:, 1]
    heterogeneous_effect = 0.2 + 0.8 * (features[:, 2] > 0) - 0.3 * (features[:, 3] > 1)
    probability = 1 / (1 + np.exp(-(base_logit + treatment * heterogeneous_effect)))
    visit = rng.binomial(1, probability)
    conversion = visit * rng.binomial(1, 0.15, rows)
    frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(12)])
    frame["treatment"] = treatment
    frame["conversion"] = conversion
    frame["visit"] = visit
    frame["exposure"] = treatment * rng.binomial(1, 0.7, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    manifest = {"source": "synthetic smoke-test data", "data_kind": "synthetic", "rows": rows}
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare(source: Path, full_path: Path, sample_path: Path, sample_rows: int, write_full: bool = True) -> dict[str, Any]:
    if sample_rows < 100:
        raise ValueError("sample_rows must be at least 100")
    if not source.is_file():
        raise FileNotFoundError(f"Data source not found: {source}. Run download first or supply --source")
    if full_path.resolve() == sample_path.resolve():
        raise ValueError("Full and sample outputs must have different paths")
    con = duckdb.connect()
    relation = relation_for(source)
    try:
        columns = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]
        required = {"treatment", "visit", "conversion", *FEATURES}
        if missing := required - set(columns):
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        bad_counts = {}
        for field in ("treatment", "visit", "conversion"):
            bad_counts[field] = int(con.execute(
                f"SELECT COUNT(*) FROM {relation} WHERE {field} IS NULL OR {field} NOT IN (0, 1)"
            ).fetchone()[0])
        if any(bad_counts.values()):
            raise ValueError(f"Invalid treatment/outcome values: {bad_counts}")
        row_count = int(con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0])
        if row_count < 100:
            raise ValueError("Data source must contain at least 100 rows")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        if write_full:
            con.execute(f"COPY (SELECT * FROM {relation}) TO '{_quote_path(full_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        selected_rows = min(sample_rows, row_count)
        con.execute(
            f"COPY (SELECT * FROM {relation} USING SAMPLE reservoir({selected_rows} ROWS) REPEATABLE (42)) TO '{_quote_path(sample_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        null_sql = ", ".join(f"COUNT(*) - COUNT({column}) AS {column}" for column in columns)
        null_values = con.execute(f"SELECT {null_sql} FROM {relation}").fetchone()
        null_counts = {column: int(value) for column, value in zip(columns, null_values)}
    finally:
        con.close()
    sidecar = source.with_suffix(source.suffix + ".manifest.json")
    source_manifest = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    return {
        "source": source_manifest.get("source", str(source)),
        "data_kind": source_manifest.get("data_kind", "unverified_external_data"),
        "rows": row_count, "columns": columns, "null_counts": null_counts,
        "model_sample_rows": selected_rows, "sample_seed": 42,
        "full_parquet_written": write_full,
        "source_sha256": source_manifest.get("sha256"),
    }


def write_report(metrics: dict[str, Any], path: Path) -> None:
    data = metrics.get("data", {})
    uplift = metrics.get("uplift", {})
    outcome = uplift.get("outcome", "visit")
    winner = uplift.get("selected_model_by_validation_separate_relative_auuc")
    best = uplift.get("selected_model_test_metrics", {})
    verified = data.get("data_kind") == "official_criteo_v2.1"
    lines = [
        "# Candidate portfolio metrics — verify before use", "",
        "> Generated from this particular pipeline execution. Results are not guaranteed to be reproducible on another environment or after code changes.",
        "", "## Dataset and experiment", "",
        f"- Data kind: `{data.get('data_kind', 'unknown')}`; source: {data.get('source', 'unknown')}",
        f"- Rows in full experiment: {metrics.get('experiment', {}).get('rows', 0):,}",
        f"- Modeling rows: {uplift.get('eligible_rows', 0):,}; validation: {uplift.get('validation_rows', 0):,}; independent test: {uplift.get('test_rows', 0):,}",
    ]
    effect = metrics.get("experiment", {}).get("outcomes", {}).get(outcome, {})
    if verified and effect and best:
        lines += [
            "", "## Candidate claims (official data; independently check before sharing)", "",
            f"- Evaluated randomized-assignment effects on **{data.get('rows', 0):,} Criteo v2.1 records**; "
            f"absolute {outcome} ITT effect **{effect['absolute_effect']:.6f}** (95% CI "
            f"[{effect['ci_95_lower']:.6f}, {effect['ci_95_upper']:.6f}]).",
            f"- Compared S-, T-, and X-learners; selected **{winner}** using validation AUUC; "
            f"its unseen-test separate-relative AUUC was **{best['separate_relative_auuc']:.6f}**.",
        ]
    else:
        lines += ["", "## No official-data résumé claims", "", "This run is synthetic or not independently provenance-verified."]
    lines += [
        "", "## Guardrails", "",
        "- Selected learner is chosen ONLY on validation, not on the independent test.",
        "- Treatment is assignment (ITT); exposure is post-treatment and not a baseline feature.",
        "- 85% expected assignment share is assumed; the pooled trials lack trial IDs.",
        "- Released Criteo data were nonuniformly subsampled; results are benchmark estimates, not a population-wide ad ROI estimate.",
        "- Constant-propensity IPW Qini and policy sums are diagnostic, not realized revenue.",
        "- An approximate post-hoc MDE is not an experimental design power guarantee.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark(full_path: Path, sample_path: Path, output_path: Path, outcome: str, expected_treatment_rate: float, budget_fraction: float, data_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if not full_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError("Prepared Parquet files missing: run prepare before benchmark")
    if not 0 < budget_fraction <= 1 or not np.isfinite(budget_fraction):
        raise ValueError("budget_fraction must be in (0, 1]")
    con = duckdb.connect()
    try:
        experiment = analyze_experiment(con, relation_for(full_path), expected_treatment_rate)
    finally:
        con.close()
    sample = pd.read_parquet(sample_path)
    uplift, predictions = fit_and_evaluate(sample, outcome, budget_fraction)
    if data_manifest and data_manifest.get("rows") != experiment["rows"]:
        raise ValueError("Manifest row count disagrees with actual full dataset; rerun prepare")
    if data_manifest and data_manifest.get("model_sample_rows") != len(sample):
        raise ValueError("Manifest modeling row count disagrees with sample; rerun prepare")
    metrics = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "environment": {"python": platform.python_version(), "platform": platform.platform(), "duckdb": duckdb.__version__}, "data": data_manifest or {"source": str(full_path), "rows": experiment["rows"], "model_sample_rows": len(sample)}, "experiment": experiment, "uplift": uplift}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    artifact_prefix = "" if output_path.stem == "metrics" else f"{output_path.stem}_"
    predictions.to_parquet(output_path.with_name(f"{artifact_prefix}holdout_uplift_predictions.parquet"), index=False)
    write_report(metrics, output_path.with_name(f"{artifact_prefix}resume_metrics.md"))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    down = sub.add_parser("download")
    down.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    down.add_argument("--i-accept-license", action="store_true")
    sample = sub.add_parser("sample")
    sample.add_argument("--source", type=Path, default=DEFAULT_SMOKE_SOURCE)
    sample.add_argument("--rows", type=int, default=50000)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prep.add_argument("--full", type=Path, default=DEFAULT_FULL)
    prep.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    prep.add_argument("--sample-rows", type=int, default=1000000)
    prep.add_argument("--skip-full-parquet", action="store_true")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--full", type=Path, default=DEFAULT_FULL)
    bench.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    bench.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    bench.add_argument("--outcome", choices=("visit", "conversion"), default="visit")
    bench.add_argument("--expected-treatment-rate", type=float, default=0.85)
    bench.add_argument("--budget-fraction", type=float, default=0.10)
    all_cmd = sub.add_parser("all")
    all_cmd.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    all_cmd.add_argument("--full", type=Path, default=DEFAULT_FULL)
    all_cmd.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    all_cmd.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    all_cmd.add_argument("--sample-rows", type=int, default=1000000)
    all_cmd.add_argument("--outcome", choices=("visit", "conversion"), default="visit")
    all_cmd.add_argument("--expected-treatment-rate", type=float, default=0.85)
    all_cmd.add_argument("--budget-fraction", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        result = download(args.source, args.i_accept_license)
    elif args.command == "sample":
        result = generate_sample(args.source, args.rows)
    elif args.command == "prepare":
        result = prepare(args.source, args.full, args.sample, args.sample_rows, not args.skip_full_parquet)
        args.sample.parent.mkdir(parents=True, exist_ok=True)
        (args.sample.parent / "data_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "benchmark":
        manifest_path = args.sample.parent / "data_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        result = benchmark(args.full, args.sample, args.output, args.outcome, args.expected_treatment_rate, args.budget_fraction, manifest)
    else:
        manifest = prepare(args.source, args.full, args.sample, args.sample_rows, True)
        args.sample.parent.mkdir(parents=True, exist_ok=True)
        (args.sample.parent / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = benchmark(args.full, args.sample, args.output, args.outcome, args.expected_treatment_rate, args.budget_fraction, manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

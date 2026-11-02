import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip('duckdb')
pytest.importorskip('pyarrow')

from marketing_uplift.pipeline import generate_sample, prepare, benchmark


def test_synthetic_end_to_end(tmp_path: Path):
    source = tmp_path / 'smoke.csv.gz'
    full = tmp_path / 'full.parquet'
    sample = tmp_path / 'sample.parquet'
    output = tmp_path / 'outputs' / 'metrics.json'
    generate_sample(source, 2000, seed=17)
    manifest = prepare(source, full, sample, sample_rows=1200)
    assert manifest['data_kind'] == 'synthetic'
    assert manifest['rows'] == 2000
    result = benchmark(full, sample, output, 'visit', 0.85, 0.10, manifest)
    assert result['experiment']['rows'] == 2000
    assert result['uplift']['validation_rows'] > 0
    assert result['uplift']['test_rows'] > 0
    assert result['uplift']['selected_model_by_validation_separate_relative_auuc']
    assert json.loads(output.read_text())['data']['data_kind'] == 'synthetic'
    assert 'No official-data' in (output.parent / 'resume_metrics.md').read_text()

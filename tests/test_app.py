"""Smoke tests for the Streamlit app.  Run:  pip install pytest && pytest -q tests/"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
BASE = pd.read_csv(ROOT / "test_data.csv")
EXPECTED = pd.read_csv(ROOT / "model" / "metrics_summary.csv", index_col=0)


def run_app(tmp_path, csv_text=None, model=None):
    app_dir = tmp_path / "app"
    shutil.copytree(ROOT, app_dir, ignore=shutil.ignore_patterns(".git", "tests", "__pycache__"))
    if csv_text is not None:
        (app_dir / "test_data.csv").write_text(csv_text)
    at = AppTest.from_file(str(app_dir / "app.py"), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Use bundled test_data.csv").run()
    if model:
        at.sidebar.selectbox[0].select(model).run()
    return at


@pytest.mark.parametrize("model", list(EXPECTED.index))
def test_metrics_match_training_report(tmp_path, model):
    at = run_app(tmp_path, model=model)
    assert not at.exception
    shown = {m.label: float(m.value) for m in at.metric[:6]}
    for metric, value in EXPECTED.loc[model].items():
        assert shown[metric] == pytest.approx(value, abs=1e-4), (model, metric)


def test_upload_without_target_shows_predictions_only(tmp_path):
    at = run_app(tmp_path, BASE.drop(columns="Revenue").to_csv(index=False))
    assert not at.exception and not at.error
    assert any("No 'Revenue' column" in w.value for w in at.warning)


def test_missing_values_are_dropped_with_warning(tmp_path):
    df = BASE.copy()
    df.loc[:4, "PageValues"] = np.nan
    at = run_app(tmp_path, df.to_csv(index=False))
    assert not at.exception
    assert any("Dropped 5 row(s)" in w.value for w in at.warning)


def test_single_class_file_does_not_crash(tmp_path):
    at = run_app(tmp_path, BASE[BASE.Revenue == False].to_csv(index=False))  # noqa: E712
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["AUC"] == "n/a"


@pytest.mark.parametrize("bad_csv", ["", "hello world\nfoo", BASE.drop(columns="Month").to_csv(index=False)])
def test_bad_files_give_a_clear_error(tmp_path, bad_csv):
    at = run_app(tmp_path, bad_csv)
    assert not at.exception
    assert at.error


def test_bundled_data_loads_by_default(tmp_path):
    app_dir = tmp_path / "app"
    shutil.copytree(ROOT, app_dir, ignore=shutil.ignore_patterns(".git", "tests", "__pycache__"))
    at = AppTest.from_file(str(app_dir / "app.py"), default_timeout=180).run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Use bundled test_data.csv"
    assert len(at.metric) >= 6


def test_upload_mode_without_file_shows_hint(tmp_path):
    at = run_app(tmp_path)
    at.sidebar.radio[0].set_value("Upload my CSV").run()
    assert not at.exception
    assert any("Upload a test CSV" in i.value for i in at.info)

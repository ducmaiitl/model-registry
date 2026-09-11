"""Client tests against a local MLflow (sqlite + local files) — no docker needed.

The behaviour under test is the decoupling contract: a consumer asks for an
alias, and what it gets back changes when someone repoints that alias, without
the consumer changing at all.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from registry import ModelRegistry  # noqa: E402


@pytest.fixture
def reg(tmp_path, monkeypatch):
    """A registry backed by a throwaway sqlite db + artifact dir."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    # Keep mlruns/ out of the repo when tests run.
    monkeypatch.chdir(tmp_path)

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    registry = ModelRegistry(tracking_uri)
    mlflow.set_experiment("test")
    return registry


@pytest.fixture
def model_dir(tmp_path):
    """A directory that looks like a set of pretrained weights."""
    d = tmp_path / "fake_model"
    d.mkdir()
    (d / "model.bin").write_bytes(b"weights")
    (d / "config.json").write_text(json.dumps({"arch": "whisper"}))
    return d


def test_register_returns_version(reg, model_dir):
    version = reg.register("m", model_dir)
    assert version == "1"
    assert isinstance(version, str)


def test_alias_decoupling(reg, model_dir):
    """The consumer call is identical before and after promotion."""
    v1 = reg.register("m", model_dir)
    v2 = reg.register("m", model_dir)

    reg.promote("m", v1, "production")
    assert reg.resolve("m", "production").version == v1

    # Deploy a new model: repoint the alias. No consumer code changes.
    reg.promote("m", v2, "production")
    assert reg.resolve("m", "production").version == v2


def test_rollback(reg, model_dir):
    v1 = reg.register("m", model_dir)
    v2 = reg.register("m", model_dir)

    reg.promote("m", v2, "production")
    reg.promote("m", v1, "production")
    assert reg.resolve("m", "production").version == v1


def test_pin_exact_version(reg, model_dir):
    reg.register("m", model_dir)
    v2 = reg.register("m", model_dir)
    reg.promote("m", v2, "production")

    pinned = reg.resolve("m", "1")
    assert pinned.version == "1"
    assert pinned.alias == ""


def test_resolve_downloads_files(reg, model_dir, tmp_path):
    v1 = reg.register("m", model_dir)
    reg.promote("m", v1, "production")

    resolved = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    assert (resolved.local_path / "model.bin").exists()
    assert (resolved.local_path / "config.json").exists()
    assert resolved.resolve_seconds > 0


def test_list_versions_and_models(reg, model_dir):
    reg.register("m", model_dir)
    reg.register("m", model_dir)

    assert "m" in reg.list_models()
    assert [v["version"] for v in reg.list_versions("m")] == ["1", "2"]


def test_list_versions_reports_aliases(reg, model_dir):
    """Regression: search_model_versions returns empty aliases, so list_versions
    has to source them from the registered model instead."""
    v1 = reg.register("m", model_dir)
    v2 = reg.register("m", model_dir)
    reg.promote("m", v1, "production")
    reg.promote("m", v2, "canary")

    by_version = {v["version"]: v["aliases"] for v in reg.list_versions("m")}
    assert by_version[v1] == ["production"]
    assert by_version[v2] == ["canary"]

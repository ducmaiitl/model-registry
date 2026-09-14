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
    # Keep mlruns/ out of the repo, and the resolve cache out of ~/.cache.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_REGISTRY_CACHE", str(tmp_path / "default-cache"))

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
    assert resolved.from_cache is False
    # Slot layout is <cache>/<name>/<version>, so alias and pin share files.
    assert resolved.local_path == tmp_path / "cache" / "m" / v1


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


def test_find_version_by_tag(reg, model_dir):
    """Importer idempotency hook: locate a version by an exact tag value."""
    v1 = reg.register("m", model_dir, tags={"hf_revision": "abc123"})
    v2 = reg.register("m", model_dir, tags={"hf_revision": "def456"})

    assert reg.find_version_by_tag("m", "hf_revision", "abc123") == v1
    assert reg.find_version_by_tag("m", "hf_revision", "def456") == v2
    assert reg.find_version_by_tag("m", "hf_revision", "nope") is None
    assert reg.find_version_by_tag("does-not-exist", "hf_revision", "abc123") is None


def test_failed_upload_leaves_no_model_shell(reg, model_dir, monkeypatch):
    """If artifact upload dies, nothing must appear in the registry. Observed
    for real: two killed imports left empty registered models behind."""
    def boom(*args, **kwargs):
        raise OSError("simulated upload failure")

    # reg._mlflow is the global mlflow module — patch via monkeypatch so the
    # stub is undone after this test instead of leaking into every later one.
    monkeypatch.setattr(reg._mlflow, "log_artifacts", boom)
    with pytest.raises(OSError):
        reg.register("never-registered", model_dir)

    assert "never-registered" not in reg.list_models()


# -- resolve cache ----------------------------------------------------------

def test_second_resolve_is_served_from_cache(reg, model_dir, tmp_path, monkeypatch):
    v1 = reg.register("m", model_dir)
    reg.promote("m", v1, "production")
    first = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    assert first.from_cache is False

    def no_download(*a, **k):
        raise AssertionError("download_artifacts must not be called on a cache hit")

    monkeypatch.setattr(reg._mlflow.artifacts, "download_artifacts", no_download)
    second = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    assert second.from_cache is True
    assert second.local_path == first.local_path
    assert (second.local_path / "model.bin").read_bytes() == b"weights"


def test_alias_and_pinned_version_share_one_cache_slot(reg, model_dir, tmp_path):
    v1 = reg.register("m", model_dir)
    reg.promote("m", v1, "production")
    by_alias = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    by_pin = reg.resolve("m", v1, cache_dir=tmp_path / "cache")
    assert by_pin.local_path == by_alias.local_path
    assert by_pin.from_cache is True


def test_incomplete_slot_is_discarded_and_refetched(reg, model_dir, tmp_path):
    """A slot without the completion marker is a half-finished download."""
    v1 = reg.register("m", model_dir)
    reg.promote("m", v1, "production")
    slot = tmp_path / "cache" / "m" / v1
    slot.mkdir(parents=True)
    (slot / "model.bin").write_bytes(b"trunc")      # partial file, no marker
    (slot / "garbage.tmp").write_bytes(b"x")

    resolved = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    assert resolved.from_cache is False
    assert resolved.local_path == slot
    assert (slot / "model.bin").read_bytes() == b"weights"
    assert not (slot / "garbage.tmp").exists()
    assert (slot / ".registry-complete").is_file()


def test_default_cache_dir_comes_from_env(reg, model_dir, tmp_path):
    v1 = reg.register("m", model_dir)
    reg.promote("m", v1, "production")
    resolved = reg.resolve("m", "production")  # no cache_dir
    assert resolved.local_path == tmp_path / "default-cache" / "m" / v1


# -- promote audit ----------------------------------------------------------

def test_promote_records_who_moved_what_from_where(reg, model_dir):
    v1 = reg.register("m", model_dir)
    v2 = reg.register("m", model_dir)

    first = reg.promote("m", v1, "production", actor="ci")
    assert first["previous_version"] is None
    assert first["version"] == v1 and first["promoted_by"] == "ci"

    second = reg.promote("m", v2, "production", actor="mike")
    assert second["previous_version"] == v1

    info = reg.alias_info("m", "production")
    assert info["version"] == v2
    assert info["previous_version"] == v1
    assert info["promoted_by"] == "mike"
    assert info["promoted_at"].endswith("+00:00")

    # The version itself remembers when it held the alias.
    tags = reg.client.get_model_version("m", v2).tags
    assert tags["promoted.production.by"] == "mike"
    assert "promoted.production.at" in tags


def test_alias_info_is_none_when_never_promoted(reg, model_dir):
    reg.register("m", model_dir)
    assert reg.alias_info("m", "production") is None

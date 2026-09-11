"""Unit tests for the pure parts of scripts/register_from_hf.py.

No network: the download path is exercised end-to-end via `--dry-run` and
real runs against MinIO, not here. These cover the logic that decides
*what* gets registered — file selection, patching, and catalog merging —
which is where a silent mistake would ship a broken model version.
"""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import register_from_hf as rfh  # noqa: E402


# -- selected() -------------------------------------------------------------

def test_selected_no_patterns_takes_everything():
    assert rfh.selected("model.safetensors", None, [])
    assert rfh.selected("README.md", [], [])


def test_selected_include_is_a_whitelist():
    include = ["*.onnx", "tokens.txt"]
    assert rfh.selected("encoder.onnx", include, [])
    assert rfh.selected("tokens.txt", include, [])
    assert not rfh.selected("model.pt", include, [])
    assert not rfh.selected("README.md", include, [])


def test_selected_exclude_wins_over_include():
    # The gipformer repo ships the ONNX files AND a 279 MB training
    # checkpoint; and mms-lid ships the same weights as .bin and .safetensors.
    assert not rfh.selected("pytorch_model.bin", None, ["*.bin"])
    assert rfh.selected("model.safetensors", None, ["*.bin"])
    assert not rfh.selected("encoder.onnx", ["*.onnx"], ["encoder.*"])


# -- apply_patches() --------------------------------------------------------

def test_apply_patches_rewrites_target(tmp_path):
    (tmp_path / "inference_wav2vec.yaml").write_text(
        "wav2vec2: !new:speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2\n"
    )
    rfh.apply_patches(tmp_path, [{
        "file": "inference_wav2vec.yaml",
        "replace": {
            "old": "speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2",
            "new": "speechbrain.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2",
        },
    }])
    assert "huggingface_transformers.wav2vec2.Wav2Vec2" in (tmp_path / "inference_wav2vec.yaml").read_text()
    assert "huggingface_wav2vec.HuggingFaceWav2Vec2" not in (tmp_path / "inference_wav2vec.yaml").read_text()


def test_apply_patches_refuses_when_target_missing(tmp_path):
    """Upstream changed the file: fail loudly rather than register an
    unpatched snapshot that only breaks at load time."""
    (tmp_path / "cfg.yaml").write_text("something else entirely\n")
    with pytest.raises(RuntimeError, match="patch target not found"):
        rfh.apply_patches(tmp_path, [{"file": "cfg.yaml", "replace": {"old": "absent", "new": "x"}}])


def test_apply_patches_noop_on_empty():
    rfh.apply_patches(Path("/nonexistent"), [])
    rfh.apply_patches(Path("/nonexistent"), None)


# -- load_catalog() ---------------------------------------------------------

def test_load_catalog_merges_defaults(tmp_path):
    cat = tmp_path / "models.yaml"
    cat.write_text(textwrap.dedent("""
        defaults:
          exclude: ["*.h5", "*.msgpack"]
          tags: {registered_by: catalog, team: stt}
        models:
          - name: a
            repo: org/a
          - name: b
            repo: org/b
            exclude: ["*.bin"]
            tags: {team: tts, framework: x}
    """))
    specs = {s["name"]: s for s in rfh.load_catalog(cat)}

    assert specs["a"]["exclude"] == ["*.h5", "*.msgpack"]
    assert specs["a"]["tags"] == {"registered_by": "catalog", "team": "stt"}

    # Entry-level values extend excludes and override tag keys.
    assert specs["b"]["exclude"] == ["*.h5", "*.msgpack", "*.bin"]
    assert specs["b"]["tags"] == {"registered_by": "catalog", "team": "tts", "framework": "x"}


def test_load_catalog_empty_file(tmp_path):
    cat = tmp_path / "models.yaml"
    cat.write_text("")
    assert rfh.load_catalog(cat) == []


# -- parse_kv() -------------------------------------------------------------

def test_parse_kv():
    assert rfh.parse_kv(["a=1", "b=x=y"]) == {"a": "1", "b": "x=y"}
    with pytest.raises(SystemExit):
        rfh.parse_kv(["novalue"])


# -- strip_download_metadata() ---------------------------------------------

def test_strip_download_metadata_removes_only_hf_cache(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"w")
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "model.bin.metadata").write_text("etag")

    rfh.strip_download_metadata(tmp_path)

    assert (tmp_path / "model.bin").exists()
    assert not (tmp_path / ".cache").exists()


def test_strip_download_metadata_noop_when_absent(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"w")
    rfh.strip_download_metadata(tmp_path)  # must not raise
    assert (tmp_path / "model.bin").exists()

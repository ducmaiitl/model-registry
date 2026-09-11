"""Example consumer: a serving process that pulls its model from the registry.

The point of this file is what it does NOT contain. There is no `import mlflow`,
no MinIO bucket name, no Postgres connection string, and no hardcoded version
number. It asks for "whisper-stt at @production" and gets back a path.

That means shipping a new model is an alias repoint in the registry — this
service picks it up on its next restart, with no code change, no rebuild, and no
redeploy. Rolling back is the same move in reverse.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from registry import ModelRegistry, ResolvedModel  # noqa: E402

MODEL_NAME = os.getenv("MODEL_NAME", "whisper-stt")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")


class ModelHandle:
    """Wraps a resolved model plus whatever engine actually loads the weights."""

    def __init__(self, resolved: ResolvedModel):
        self.resolved = resolved
        # resolve() downloads the version's artifact directory itself, so
        # local_path already *is* the weights dir — model.bin and config.json
        # sit directly in it. Do not append "model" here.
        self.weights_path = resolved.local_path
        self.engine = self._load_engine()

    def _load_engine(self):
        """Load the real inference engine. Placeholder for now.

        With faster-whisper this would be:

            from faster_whisper import WhisperModel
            return WhisperModel(str(self.weights_path), device="cuda")

        The registry does not care which engine this is — it only supplies files.
        """
        return None

    def transcribe(self, audio_path: str) -> str:
        if self.engine is None:
            return f"[placeholder] would transcribe {audio_path}"
        segments, _ = self.engine.transcribe(audio_path)
        return " ".join(segment.text for segment in segments)


def load_model() -> ModelHandle:
    """The one call a consumer makes."""
    reg = ModelRegistry()
    resolved = reg.resolve(MODEL_NAME, MODEL_STAGE)
    print(
        f"loaded {resolved.name} v{resolved.version} "
        f"(@{resolved.alias or 'pinned'}) in {resolved.resolve_seconds:.2f}s"
    )
    print(f"  weights: {resolved.local_path}")
    if resolved.metrics:
        print(f"  metrics: {resolved.metrics}")
    return ModelHandle(resolved)


# --------------------------------------------------------------------------
# FastAPI wiring. Uncomment to run as a real service:
#   pip install fastapi uvicorn
#   uvicorn examples.serving_fastapi:app --port 8000
# --------------------------------------------------------------------------
#
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
#
# STATE: dict[str, ModelHandle] = {}
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Resolve once at startup, not per request.
#     STATE["model"] = load_model()
#     yield
#     STATE.clear()
#
# app = FastAPI(title="STT gateway", lifespan=lifespan)
#
# @app.get("/health")
# def health():
#     return {"status": "ok", "model_loaded": "model" in STATE}
#
# @app.get("/model-info")
# def model_info():
#     r = STATE["model"].resolved
#     return {
#         "name": r.name,
#         "version": r.version,
#         "alias": r.alias,
#         "resolve_seconds": r.resolve_seconds,
#         "metrics": r.metrics,
#     }
#
# @app.post("/transcribe")
# def transcribe(audio_path: str):
#     return {"text": STATE["model"].transcribe(audio_path)}


if __name__ == "__main__":
    # Smoke test: resolve and print, no server required.
    handle = load_model()
    print(f"  engine: {handle.engine or 'placeholder (no engine wired up)'}")

"""Facade over MLflow for the model registry.

Consumers import ``ModelRegistry`` from this package and nothing else. They never
import mlflow, never talk to MinIO or Postgres, and never learn which concrete
version is serving traffic. That indirection is the whole point: the model
lifecycle moves independently of the code that consumes models.

The single consumer entrypoint is :meth:`ModelRegistry.resolve`, which returns a
:class:`ResolvedModel` whose ``local_path`` already has the weights on disk.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_TRACKING_URI = "http://localhost:5000"

# Artifacts are logged under this subdirectory of the run, so a model version
# points at ``runs:/<run_id>/model`` rather than the run root.
_ARTIFACT_PATH = "model"


@dataclass
class ResolvedModel:
    """What a consumer gets back from :meth:`ModelRegistry.resolve`.

    ``version`` is always a string. MLflow hands back an int on some paths, and
    string-concatenating it (``"v" + version``) blows up at runtime, so the
    cast happens here once instead of at every call site.
    """

    name: str
    version: str
    alias: str
    local_path: Path
    run_id: Optional[str] = None
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    resolve_seconds: float = 0.0


class ModelRegistry:
    """Versioned model storage with movable aliases.

    Aliases, not stages: MLflow deprecated the stage API in 2.9 and it behaves
    inconsistently, so promotion is always "point an alias at a version".
    """

    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI
        )

        # Imported lazily so that merely importing this module does not pay the
        # (substantial) mlflow import cost, and so consumers can depend on the
        # package without mlflow resolved at import time.
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow = mlflow
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    # -- write path ----------------------------------------------------------

    def register(
        self,
        name: str,
        source_dir: str | Path,
        flavor: str = "artifact",
        metrics: dict[str, float] | None = None,
        params: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
    ) -> str:
        """Register a directory of model files as a new immutable version.

        Works the same whether ``source_dir`` holds pretrained weights (today,
        with no metrics) or the output of a finetuning run (later, with real
        metrics). Returns the new version as a string.
        """
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"source_dir does not exist: {source}")

        try:
            self.client.create_registered_model(name)
        except Exception:
            # Already registered; new versions just get appended to it.
            pass

        run_tags = {"registry.flavor": flavor, "registry.model_name": name}
        with self._mlflow.start_run(run_name=f"register-{name}", tags=run_tags) as run:
            run_id = run.info.run_id
            if params:
                self._mlflow.log_params(params)
            if metrics:
                self._mlflow.log_metrics(metrics)
            self._mlflow.log_artifacts(str(source), artifact_path=_ARTIFACT_PATH)

        mv = self.client.create_model_version(
            name=name,
            source=f"runs:/{run_id}/{_ARTIFACT_PATH}",
            run_id=run_id,
        )

        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, str(mv.version), key, str(value))

        if description:
            self.client.update_model_version(
                name=name, version=str(mv.version), description=description
            )

        return str(mv.version)

    def promote(self, name: str, version: str, alias: str = "production") -> None:
        """Point ``alias`` at ``version`` — the lever that decouples deploy from code.

        Consumers ask for ``production``; this call decides what production means.
        Rolling back is the same call aimed at the previous version, so a bad
        release is undone without touching, rebuilding, or redeploying any
        consumer.
        """
        alias = alias.lower()
        self.client.set_registered_model_alias(name, alias, str(version))

    # -- read path -----------------------------------------------------------

    def resolve(
        self,
        name: str,
        ref: str = "production",
        cache_dir: str | Path | None = None,
    ) -> ResolvedModel:
        """Resolve ``name`` at ``ref`` and download its weights to disk.

        ``ref`` is either an alias (``production``, ``canary``) or an exact
        numeric version (``"3"``) for pinning. This is the only call a consumer
        needs to make.
        """
        started = time.perf_counter()

        if str(ref).isdigit():
            version = str(ref)
            alias = ""
            mv = self.client.get_model_version(name, version)
            uri = f"models:/{name}/{version}"
        else:
            alias = str(ref).lower()
            mv = self.client.get_model_version_by_alias(name, alias)
            version = str(mv.version)
            # Alias downloads use '@', not '/'. The '/' form is read as a
            # version number and fails.
            uri = f"models:/{name}@{alias}"

        dst = str(cache_dir) if cache_dir else None
        local_path = self._mlflow.artifacts.download_artifacts(
            artifact_uri=uri, dst_path=dst
        )

        run_id = mv.run_id or None
        metrics: dict[str, float] = {}
        tags: dict[str, str] = dict(mv.tags or {})
        if run_id:
            try:
                run = self.client.get_run(run_id)
                metrics = dict(run.data.metrics or {})
                for key, value in (run.data.tags or {}).items():
                    tags.setdefault(key, value)
            except Exception:
                # Run may have been deleted; the version and its files are
                # still perfectly usable, so this is not fatal.
                pass

        return ResolvedModel(
            name=name,
            version=version,
            alias=alias,
            local_path=Path(local_path),
            run_id=run_id,
            metrics=metrics,
            tags=tags,
            resolve_seconds=time.perf_counter() - started,
        )

    def list_versions(self, name: str) -> list[dict]:
        """All versions of ``name``, oldest first. Versions come back as strings."""
        versions = self.client.search_model_versions(f"name='{name}'")

        # search_model_versions does NOT populate mv.aliases (it comes back
        # empty even for aliased versions), so build the mapping from the
        # registered model, which stores it as {alias: version}.
        aliases_by_version: dict[str, list[str]] = {}
        try:
            rm = self.client.get_registered_model(name)
            for alias, version in (rm.aliases or {}).items():
                aliases_by_version.setdefault(str(version), []).append(alias)
        except Exception:
            pass

        out = [
            {
                "version": str(mv.version),
                "aliases": sorted(aliases_by_version.get(str(mv.version), [])),
                "run_id": mv.run_id,
                "status": mv.status,
                "created": mv.creation_timestamp,
            }
            for mv in versions
        ]
        out.sort(key=lambda item: int(item["version"]))
        return out

    def list_models(self) -> list[str]:
        """Names of every registered model."""
        return [rm.name for rm in self.client.search_registered_models()]

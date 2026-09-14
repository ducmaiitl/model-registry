# Model Registry Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the registry from "installable library with a pinned catalog" (v0.1.0) to the thing every model in the system flows through: consumers load from it, the benchmark scores from it, finetunes register into it, and every alias move is visible.

**Architecture:** Nothing structural changes. Every phase adds methods to the existing `ModelRegistry` facade or adds a consumer of it. The MLflow server, Postgres, GCS and the `models.yaml` catalog stay as specified in the reference spec. Two constraints hold throughout: consumers keep importing only `registry` (never `mlflow`), and all code here is unit-tested against the sqlite fixture — nothing in this plan runs a model or downloads weights on a developer machine.

**Tech Stack:** Python 3.11, mlflow-skinny (client) / mlflow 2.16.2 (server), pytest, huggingface_hub, jiwer + a Vietnamese text normalizer (Phase 2), Google Speech-to-Text v2 client (Phase 2, GCP baseline only).

**Reference spec:** `docs/superpowers/specs/2026-09-14-model-registry-design.md` (this repo). robo-be context: `robo-be/docs/STT-system-architecture.md`, `robo-be/docker-compose.yml`.

**Order of landing:** each phase is independently committable and leaves everything working.

- Phase 1 (Tasks 1–6): **robo-be loads from the registry.** One worker at a time, one-line loader swaps, persistent cache volume, `HF_TOKEN` leaves runtime. Fixes the latent gipformer outage on the way. *Lives in the robo-be repo.*
- Phase 2 (Tasks 7–12): **Tier 1 bench as the first external consumer.** New repo `stt-tts-bench`: registry adapter, GCP Chirp 3 adapter, one normalizer, jiwer, results logged to the same MLflow. The GCP-vs-open-source comparison that has never been run.
- Phase 3 (Tasks 13–15): **Observability emission.** Metrics hook in the facade, `REGISTRY_ACTOR` from CI, alias-change notification.
- Phase 4 (Tasks 16–21): **Finetune readiness.** `best_version`, `promote_if_better`, dataset versioning, training-run context, `base_model = name@version` convention. Land *before* the first finetune job exists.
- Phase 5 (Tasks 22–24): **Hygiene.** Orphan-run GC, cache prune, catalog-vs-registry verify.

Every task has a commit. Phases 3–5 are in this repo on `main`; Phase 1 is in robo-be; Phase 2 is a new repo.

---

## File structure (reference — do not create in one sweep)

**This repo — new:**
- `src/registry/metrics.py` (Phase 3: `MetricsHook` protocol + no-op default)
- `src/registry/training.py` (Phase 4: `TrainingRun` context manager)
- `src/registry/datasets.py` (Phase 4: `register_dataset`, `dataset_hash`)
- `scripts/registry_gc.py` (Phase 5) — or subcommands on `registry_cli.py`
- `tests/test_metrics_hook.py`, `tests/test_training.py`, `tests/test_datasets.py`, `tests/test_gc.py`

**This repo — modified:**
- `src/registry/client.py` (Phases 3, 4, 5: hook wiring, `best_version`, `promote_if_better`, `prune_cache`, `gc_orphan_runs`, `verify_catalog`)
- `scripts/registry_cli.py` (new subcommands: `best`, `gc`, `cache prune`, `catalog verify`)
- `docs/OBSERVABILITY.md`, `README.md`, `.github/workflows/ci.yml` (set `REGISTRY_ACTOR`)

**robo-be — modified (Phase 1):**
- `robo_infer/config.py` (add `registry_tracking_uri`, `registry_cache_dir`, per-model `*_ref` settings)
- `robo_infer/models/asr_cohere.py`, `asr_qwen.py`, `asr_gipformer.py`, `asr_funasr.py`, `lid_router/{mms_lid,lid_classifier,phoneme_lid}.py`, `voxcpm_tts.py`
- `docker-compose.yml` (registry env + cache volume; drop `HF_TOKEN` from runtime services)
- `Dockerfile.lid` (drop the bake step), `requirements.*.txt` (add `robo-model-registry`)

**stt-tts-bench — new repo (Phase 2):** see Task 7.

---

## Phase 1 — robo-be loads from the registry

*Repo: robo-be. Code changes only; validation runs where the GPU is. Each task swaps one worker and is a separate commit so a regression is bisectable.*

### Task 1: Add the registry client and settings to robo-be

- [ ] Add `robo-model-registry @ git+https://github.com/ducmaiitl/model-registry@main` to every `requirements.<worker>.txt` that loads a model (lid, qwen, cohere, gipformer, voxcpm).
- [ ] `robo_infer/config.py`: add
  ```python
  registry_tracking_uri: str = "http://localhost:5000"   # ROBO_REGISTRY_TRACKING_URI
  registry_cache_dir: str = "/models"                     # ROBO_REGISTRY_CACHE_DIR
  ```
  and one `<model>_ref: str = "production"` per model (e.g. `cohere_ref`, `gipformer_ref`), so an operator can pin a worker to `"3"` without a rebuild.
- [ ] `docker-compose.yml`: add a named volume `registry_cache:/models` to every model worker; set `ROBO_REGISTRY_TRACKING_URI` and `MODEL_REGISTRY_CACHE=/models` in each worker's env block.
- [ ] Unit test: `Settings()` parses the new fields with defaults.
- [ ] Commit: `feat(infer): registry client dependency, settings, cache volume`.

### Task 2: Cohere sidecar → registry

- [ ] `asr_cohere.py`: replace `_HF_REPO` usage in `load()` with
  ```python
  files = ModelRegistry(settings.registry_tracking_uri).resolve("cohere-asr", settings.cohere_ref).local_path
  self._processor = AutoProcessor.from_pretrained(files)
  self._model = CohereAsrForConditionalGeneration.from_pretrained(files, torch_dtype=torch.bfloat16, device_map="auto")
  ```
  Keep `_HF_REPO` only in the log line for now (`"Loading cohere-asr@%s (v%s)"` with `resolved.version`).
- [ ] Log `resolved.version`, `resolved.from_cache`, `resolved.resolve_seconds` at INFO on load — this is the observability seam.
- [ ] Unit test with a fake `ModelRegistry` (dependency-inject via a module-level factory) asserting `from_pretrained` receives a directory path, not a repo id.
- [ ] Remove `HF_TOKEN` from the `robo-infer-cohere` compose env block.
- [ ] Commit: `feat(cohere): load weights from the model registry`.

### Task 3: Gipformer sidecar → registry (fixes the latent outage)

- [ ] `asr_gipformer.py`: replace the four `hf_hub_download` calls with `files / name` for the four filenames; keep `_FILES_FP32` / `_FILES_INT8` maps.
- [ ] The registry version is pinned at `ba00dad1` (old filenames), so the current maps work unchanged. **Do not** bump the catalog pin until this task lands; then bump pin + filename maps together in a follow-up commit.
- [ ] Delete the dead top-of-file `hf_hub_download` block (lines 14–17) that references filenames nothing uses.
- [ ] Unit test: recognizer receives paths under the resolved directory.
- [ ] Commit: `fix(gipformer): load ONNX bundle from the registry instead of HF HEAD`.

### Task 4: Qwen and Fun-ASR sidecars → registry

- [ ] `asr_qwen.py`: `Qwen3ASRModel.from_pretrained(files, ...)`; `_QWEN_DEFAULT` remains only as the catalog `repo`.
- [ ] `asr_funasr.py`: `AutoModel(model=str(files), ...)`. Verify funasr resolves the `Qwen3-0.6B/` sub-tokenizer from a local dir (it is inside the snapshot).
- [ ] Tests as in Task 2.
- [ ] Commit: `feat(asr): qwen and funasr sidecars load from the registry`.

### Task 5: LID worker → registry; delete the bake step

- [ ] `lid_router/mms_lid.py`: `from_pretrained(files)` for `mms-lid-256`.
- [ ] `lid_router/phoneme_lid.py`: `from_pretrained(files)` for `xlsr53-phoneme`.
- [ ] `lid_router/lid_classifier.py`: `foreign_class(source=str(files), ...)` for `voxlingua107-lid`. The yaml patch is already applied inside the registered version.
- [ ] `Dockerfile.lid`: remove the `snapshot_download` + yaml-patch `RUN` block and the `/opt/hf_cache` bake; drop `HF_HOME`. Image shrinks by ~2.5 GB.
- [ ] `docker-compose.yml`: remove the comment forbidding a cache volume on this worker (the reason no longer exists) and add `registry_cache:/models` like the others; remove `HF_TOKEN`.
- [ ] Tests: each wrapper receives a directory.
- [ ] Commit: `feat(lid): load all three LID models from the registry; drop image bake`.

### Task 6: VoxCPM2 TTS → registry; sweep

- [ ] `voxcpm_tts.py`: resolve `voxcpm2-tts` and pass `local_path` to the voxcpm loader; `voxcpm_repo_id` in `config.py` becomes unused — delete it.
- [ ] Grep robo-be for `_HF_REPO`, `hf_hub_download`, `snapshot_download`, `HF_TOKEN`: none should remain in runtime code (benchmarks/scripts may keep them).
- [ ] Update `docs/STT-system-architecture.md` "Deployment" table: weights come from the registry; note `ROBO_*_REF` pinning.
- [ ] Commit: `feat(tts): voxcpm loads from the registry; remove last HF Hub runtime references`.

---

## Phase 2 — Tier 1 bench as the first external consumer

*New repo `stt-tts-bench`. Raw model quality, offline: audio file in → text out. No wire protocol. Runs on a GPU machine; unit tests run anywhere.*

### Task 7: Repo skeleton and the backend seam

- [ ] Layout:
  ```
  stt-tts-bench/
  ├── pyproject.toml            deps: robo-model-registry, jiwer, pyyaml, mlflow-skinny; extras: gcp, torch
  ├── bench/
  │   ├── backends/__init__.py  Protocol: transcribe(audio_path) -> Hypothesis(text, seconds)
  │   ├── backends/registry_hf.py   resolve(name, ref) → transformers pipeline on local_path
  │   ├── backends/gcp_chirp3.py    Speech v2 recognize (batch, not streaming)
  │   ├── normalize.py          normalize_vi(text): NFC, lowercase, strip punct, digits→words
  │   ├── score.py              wer/cer via jiwer on normalized pairs; code-switch pass
  │   ├── dataset.py            manifest.csv: path, ref, speaker, lang, noise, duration
  │   └── run.py                CLI: --dataset --backend ... → report.md + MLflow run
  └── tests/                    normalizer, scorer, manifest parsing, fake backend
  ```
- [ ] `backends/__init__.py`: `class Backend(Protocol): name: str; def transcribe(self, path: Path) -> Hypothesis`.
- [ ] Commit: `feat: bench skeleton with backend protocol`.

### Task 8: One normalizer, tested against known traps

- [ ] `normalize_vi`: `unicodedata.normalize("NFC")`, casefold, strip punctuation, collapse whitespace, digits → Vietnamese words (`4` → `bốn`) via a small table for 0–100.
- [ ] Tests: NFD vs NFC input normalize equal; `"Bốn câu"` == `"4 câu"`; `"CÁI KHÓ LÓ CÁI KHÔN"` == lowercase ref; punctuation-only differences score 0 WER.
- [ ] `score.py`: `wer`, `cer` via `jiwer`; `code_switch_preserved(hyp, en_words)`; a version string `NORMALIZER_VERSION = "1"` logged as a run param — a normalizer change shifts every historical number.
- [ ] Commit: `feat: Vietnamese normalizer and jiwer scorer`.

### Task 9: Dataset manifest with slices

- [ ] `dataset.py`: read `manifest.csv` with columns `id, path, ref, speaker(child|adult|synthetic), lang(vi|en|cs), noise(clean|noisy), duration_s, en_words`.
- [ ] Recover robo-be's ground truth (`refs.csv`, `refs-mix.csv`, `expected.json`) into one manifest; mark t204's known-bad reference; tag t2xx as `adult`, ElevenLabs as `synthetic`.
- [ ] Report groups by every slice column. The `child` slice is empty until recordings exist — the report must say so explicitly rather than silently averaging adults.
- [ ] Commit: `feat: dataset manifest with speaker/lang/noise slices`.

### Task 10: Registry backend

- [ ] `backends/registry_hf.py`: `RegistryBackend(name, ref, engine="transformers")` → `ModelRegistry().resolve(name, ref)`; build the engine from `local_path` per `framework` tag (`transformers` → `AutoProcessor`/`AutoModel…`; `sherpa-onnx` → recognizer from the four files). Log `resolved.version`, `hf_revision` tag, `from_cache` into the run params.
- [ ] Unit test with a fake registry returning a temp dir and a stub engine.
- [ ] Commit: `feat: registry-backed candidates`.

### Task 11: GCP Chirp 3 baseline backend

- [ ] `backends/gcp_chirp3.py`: Speech-to-Text v2 `recognize` on the whole file (`chirp_3`, region `us`, language codes vi-VN + en-US). Record `$/min` from the published price as a run param so cost appears in every report.
- [ ] Unit test with the client stubbed.
- [ ] Commit: `feat: GCP Chirp 3 baseline backend`.

### Task 12: Runner, report, MLflow logging, success criteria

- [ ] `run.py`: for each backend × dataset → per-sample hyps, sliced WER/CER/CS-pass, latency p50/p95; write `report.md` with a **Chirp 3 row always present**; log one MLflow run per backend with params `{backend, model, version, hf_revision, dataset_version, normalizer_version}` and the metrics.
- [ ] Encode the migration criteria in `criteria.yaml` and evaluate them in the report: `child-slice WER ≤ chirp3 + 3pp`, `cs_pass ≥ chirp3`, `cost < chirp3`. Placeholders until agreed; the point is deciding before seeing numbers.
- [ ] Commit: `feat: runner with sliced report and MLflow logging`.

---

## Phase 3 — Observability emission

### Task 13: Metrics hook in the facade

- [ ] `src/registry/metrics.py`: `class MetricsHook(Protocol): def resolve(self, name, version, alias, seconds, from_cache, ok: bool)`, `def promote(self, record)`; `NoopHook` default.
- [ ] `ModelRegistry(tracking_uri=None, metrics=None)`; `resolve()` and `promote()` call the hook in a `finally` so failures are counted too.
- [ ] Tests: hook receives one call per resolve with correct `from_cache`; receives failures.
- [ ] Commit: `feat: optional metrics hook on resolve/promote`.

### Task 14: Actor from CI

- [ ] `.github/workflows/ci.yml`: `env: REGISTRY_ACTOR: github-actions/${{ github.actor }}` on every job so any promotion from CI is attributed to the pipeline.
- [ ] README: document that promotions should run from CI with the actor set.
- [ ] Commit: `chore(ci): set REGISTRY_ACTOR`.

### Task 15: Alias-change notification

- [ ] `promote()` gains `notify: Callable[[dict], None] | None`; `registry_cli promote --notify-webhook URL` posts the record as JSON. No new dependency (`urllib`).
- [ ] Test with a local HTTP server fixture.
- [ ] Commit: `feat: notify on alias move`.

---

## Phase 4 — Finetune readiness

*Land before the first finetune job exists.*

### Task 16: `best_version`

- [ ] `best_version(name, metric, lower_is_better=True, require_tag=None) -> str | None`: over `list_versions`, read each run's metrics, return the best `str` version. Skips versions without the metric.
- [ ] Tests: two versions with `wer`; lower wins; missing metric skipped; `None` if none.
- [ ] Commit: `feat: best_version by metric`.

### Task 17: `promote_if_better`

- [ ] `promote_if_better(name, candidate, metric, alias="production", lower_is_better=True, min_delta=0.0) -> dict`: compare candidate's metric with the current alias target's; promote only if better by `min_delta`; return `{promoted: bool, reason, current, candidate, metric values}`.
- [ ] Tests: better → promoted with audit record; worse → not promoted, alias unchanged; alias absent → promoted.
- [ ] `registry_cli promote --if-better wer`.
- [ ] Commit: `feat: promote_if_better policy gate`.

### Task 18: Dataset versioning

- [ ] `src/registry/datasets.py`: `dataset_hash(manifest_path) -> "sha256:…"` over the manifest bytes plus each referenced file's size+mtime (content hash optional flag); `register_dataset(name, manifest_dir, hash) -> version` reusing `register()` with `flavor="dataset"` and tags `{kind: dataset, dataset_hash}`.
- [ ] Convention: a finetune's version tag `dataset_hash` must match a registered dataset version's tag; `find_version_by_tag(dataset_name, "dataset_hash", h)` looks it up.
- [ ] Tests: hash stable across runs, changes when a file changes; dataset registers and is findable.
- [ ] Commit: `feat: dataset versioning through the same registry`.

### Task 19: `TrainingRun` context

- [ ] `src/registry/training.py`:
  ```python
  with reg.training_run(name="cohere-asr", base="cohere-asr@production", dataset="child-speech@3") as run:
      run.log_params({...}); run.log_metric("wer", 0.09, step=1000)
      version = run.register(output_dir, metrics={"wer": 0.081})
  ```
  Internally one MLflow run; `register()` reuses it (no second run) and sets `base_model = "cohere-asr@<resolved version>"`, `dataset_hash`, `git_commit` (from env `GIT_COMMIT` or `git rev-parse`).
- [ ] Tests against the sqlite fixture: params/metrics land on the run; version tags set; `base_model` is `name@version`.
- [ ] Commit: `feat: TrainingRun keeps trainers behind the facade`.

### Task 20: Lineage query

- [ ] `lineage(name, version) -> dict`: `{base_model, dataset_hash, git_commit, hf_repo/hf_revision if imported, run params}` walking `base_model` recursively to the pretrained root.
- [ ] `registry_cli lineage --name --version`.
- [ ] Tests: finetune → pretrained chain resolves two levels.
- [ ] Commit: `feat: lineage query`.

### Task 21: Docs

- [ ] README "Extending to finetuning" becomes the real workflow: dataset → `training_run` → `best_version` → `promote_if_better`.
- [ ] `OBSERVABILITY.md` §6 tags table: `base_model` semantics for finetunes.
- [ ] Commit: `docs: finetune workflow`.

---

## Phase 5 — Hygiene

### Task 22: Orphan-run GC

- [ ] `gc_orphan_runs(dry_run=True) -> list[run_id]`: runs in the registry experiment that no model version references and that are `FINISHED` or older than N hours if `RUNNING`; delete via `client.delete_run` then document `mlflow gc --backend-store-uri … --artifacts-destination …` for the bytes.
- [ ] `registry_cli gc [--apply]`.
- [ ] Tests: linked run kept; orphan listed; deleted only with `apply`.
- [ ] Commit: `feat: orphan run gc`.

### Task 23: Cache prune

- [ ] `prune_cache(keep_aliased=True, keep_recent=2, cache_dir=None) -> list[Path]`: per model name, keep slots for currently aliased versions plus the N most recent by marker `resolved_at`; remove the rest and any `.partial-*` older than a day.
- [ ] `registry_cli cache prune [--apply]`.
- [ ] Tests on a temp cache tree.
- [ ] Commit: `feat: cache prune`.

### Task 24: Catalog verify

- [ ] `verify_catalog(path) -> report`: for each entry, is a version with that `hf_revision` registered? does the alias point where the catalog says (informational, since aliases may legitimately drift)? any registered model not in the catalog?
- [ ] `registry_cli catalog verify`; CI runs it in `--dry-run` form against the fixture, not the live server.
- [ ] Commit: `feat: catalog verify`.

---

## Risks

- **Loader quirks per engine (Phase 1).** funasr and voxcpm accept local directories, but each has its own way of locating sub-components. Validate one worker at a time on the GPU host; keep `_HF_REPO` reachable via an env flag for one release as a fallback.
- **Cache volume sizing (Phase 1).** All eight models ≈ 21 GB; the shared `registry_cache` volume needs headroom for two versions each. Phase 5 prune exists for this.
- **Child-speech slice is empty (Phase 2).** Every WER number is on adults until recordings with consent exist. The report must refuse to present an adult average as the migration answer.
- **Normalizer drift (Phase 2).** Changing `normalize_vi` invalidates comparisons; the version param is the guard, and reports must pin it.
- **Hook overhead (Phase 3).** Keep the hook synchronous and trivial; anything heavier belongs in the consumer.

## Out of scope (explicitly)

- Where the MLflow server runs, its auth, backups, networking.
- Streaming/wire-path benchmarking (Tier 2 stays in `robo-be/benchmarks`).
- Training code itself; only registration of its outputs.
- Multi-region or HA for the registry.

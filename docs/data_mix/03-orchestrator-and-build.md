# 03 — Orchestrator + build pipeline

## TL;DR

- `build_mix.sh` and `mix.py` assemble the source buckets into one shuffled training JSONL file.
- The build also writes separate validation files so plant, general, refusal, and persona behavior can be checked independently.
- Storage locations come from environment variables, which keeps machine-specific paths out of the public docs and code.
- Re-running with the same seed produces the same output and avoids rewriting images that already exist.

How `build_mix.sh` + `mix.py` turn the bucket builders into a single
JSONL drop-in. Covers env-var-driven storage roots, idempotence,
preflight checks, and validation.

## Storage roots — env-var indirection

All storage is configured via env vars (resolved by
`src/data_mix/src/env_paths.py`). No per-machine absolute
paths are committed to the public repo.

| Env var | Purpose | Default if unset |
|---|---|---|
| `HF_HOME` | HuggingFace datasets / hub cache | `huggingface_hub` default (`~/.cache/huggingface`) |
| `DATA_MIX_IMAGE_ROOT` | Downloaded + resized images for LLaVA / Negative / dummy | `data_mix/_local/images/` (repo-relative, gitignored) |
| `DATA_MIX_OUTPUT_ROOT` | `train.jsonl`, `val_*.jsonl`, `build_report.json` | `finetune/data/mix-<N>/` (repo-relative) |
| `PLANTNET_JSONL` | Path to the PlantNet enriched train JSONL to subsample from | `finetune/data/english-desc-v2/train.jsonl` |
| `OFFLINE_QA_JSON` (v3) | Path to the persona corpus | `assets/data_offline_qa/offline_qa.json` |
| `PYTHON_BIN` | Python interpreter | system `python` |

Machine-specific mappings (e.g. routing all four roots to a particular
mount or external drive) live in the operator's local run notes, not
in the public repo.

## On-disk layout under the configured roots

```
${HF_HOME}/                                    # HF cache (datasets, hub)
${DATA_MIX_IMAGE_ROOT}/
├── llava/                                     # 15K resized 960×672 .jpg (mix-50k)
├── negative/                                  # 5K  resized 960×672 .jpg (mix-50k)
└── dummy_gray_960x672.jpg                     # v1 only; v2 leaves smoltalk as image=None
${DATA_MIX_OUTPUT_ROOT}/
├── train.jsonl                                # ~50K rows, shuffled, mixed
├── val_plant.jsonl                            # v2: per-source val split
├── val_nonplant.jsonl                         # v2: LLaVA + smoltalk
├── val_negative.jsonl                         # v2: negative-only
├── val_offline_qa.jsonl                       # v3: persona-only
└── build_report.json                          # per-source counts, seed, dataset hashes
```

PlantNet image paths in the mix JSONL are absolute references back to
the already-resized files under `PLANTNET_JSONL`'s
`images_resized/train/` tree — the mix builder copies the absolute
paths verbatim, no re-resize, no copy.

## Multi-val output (v2)

Single-val v1 (`val.jsonl`) became multi-val v2 (`val_<source>.jsonl`)
in commit `f6d0c1f`. The motivation: catastrophic-forget watch.
Splitting val by source lets the trainer report
`eval_plant_loss` / `eval_nonplant_loss` / `eval_negative_loss` /
`eval_offline_qa_loss` per checkpoint instead of a single mean —
domain collapse shows up as one bucket's loss falling while another's
rises.

The finetune-side consumer is `cfg.data.val_files` (dict of
`{source: path}`) which threads through to `SFTTrainer.eval_dataset`
as a dict. See `finetune/configs/plantnet-50k-mix-*.yaml` for the
production wiring.

## `build_mix.sh` — the driver

Single-shot script that:

1. Resolves `HF_HOME`, `DATA_MIX_IMAGE_ROOT`, `DATA_MIX_OUTPUT_ROOT`,
   `PLANTNET_JSONL`, `OFFLINE_QA_JSON` against the user's env (with
   the documented fallbacks). Prints the resolved table before doing
   any work.
2. Verifies `PLANTNET_JSONL` exists; errors with an actionable
   message if not (commit `e3fb96d` — explicit `InsufficientPoolError`
   instead of an `assert` that disappears under `python -O`).
3. Creates `DATA_MIX_IMAGE_ROOT` and `DATA_MIX_OUTPUT_ROOT` if
   missing.
4. Calls `python -m data_mix.src.mix --config <config.yaml>`.
5. Prints the path to `build_report.json` on success.

The script is **resumable**: HF datasets caches under `HF_HOME`, and
the resized-image writer has a skip-if-exists fast path with tmp-file
cleanup on error (commit `6afa339`). Re-running after a partial run
skips already-downloaded shards + already-resized images.

## `mix.py` — the orchestrator

Reads the config YAML, calls each bucket sampler, validates the
resulting records, and writes the train+val JSONLs:

```python
def build_mix(config_path: Path) -> BuildReport:
    cfg = load_config(config_path)
    env = resolve_env_paths()

    plant_records    = sample_plant(cfg.plant, env.plantnet_jsonl)
    llava_records    = sample_llava(cfg.llava, env)
    smoltalk_records = sample_smoltalk(cfg.smoltalk, env)
    negative_records = sample_negative(cfg.negative, env)
    offline_qa       = sample_offline_qa(cfg.offline_qa, env)  # v3

    train, val_by_source = _split_train_val(
        [plant_records, llava_records, smoltalk_records, negative_records, offline_qa],
        seed=cfg.seed,
    )
    write_jsonl(env.output_root / "train.jsonl", train)
    for source, records in val_by_source.items():
        write_jsonl(env.output_root / f"val_{source}.jsonl", records)

    return BuildReport(...)  # → build_report.json
```

`_split_train_val` raises `InsufficientPoolError` (not `assert`) when
a bucket can't satisfy its train+val target after caps and filters
(commit `e3fb96d`).

## Idempotence guarantees

The Python entry point is **deterministic** in `--seed`:

- Same config + same seed + same HF dataset version → byte-identical
  `train.jsonl` + `val_*.jsonl` (commit `07f7e1c` pinned this with a
  two-run byte-equality test).
- HF dataset version drift is the one variable not controlled — the
  `build_report.json` records the commit hash where the HF datasets
  API exposes it.

The image-side helpers are also idempotent:

- `_persist_image(src, dst)` short-circuits when `dst.exists()` and
  cleans up its tmp file on any error (commit `6afa339`). Pinned by
  `test_image_resize.test_persist_image_idempotent_when_target_exists`.

## Validation tests

`src/data_mix/tests/` has 96 / 96 green tests covering:

| Layer | Tests |
|---|---|
| Schema validator (`test_schema.py`) | Role alternation, required fields, image type. |
| Dummy image (`test_dummy_image.py`) | Idempotent generation, correct shape, channel values. |
| Image resize (`test_image_resize.py`) | 960×672 stretch matches `prepare_plantnet`. Idempotent persist. |
| Plant sampler (`test_plant_sampler.py`) | Per-class cap, prompt-variant distribution, dual-source. |
| LLaVA sampler (`test_llava_sampler.py`) | Word-boundary regex, image persist, multi-turn truncation. |
| Smoltalk sampler (`test_smoltalk_sampler.py`) | text-only, deterministic shuffle. |
| Negative builder (`test_negative_builder.py`) | Refusal template, non-plant pool sampling. |
| offline_qa (`test_offline_qa_sampler.py`, v3) | Persona corpus loading + no-oversample contract (15 tests). |
| Integration (`test_mix_integration.py`) | Determinism, finetune-side drop-in, offline_qa inclusion/exclusion. |

The full suite runs against **mocked HF streams** — no network
required. Real-network runs are gated by a separate
`-m needs_network` marker (not currently enabled by default).

## End-to-end sanity (real network)

Smoke run (200 records, ~3-5 minutes once HF cache is warm):

```bash
cd <repo>
export CONFIG=$PWD/src/data_mix/configs/mix-200-llava.yaml
bash src/data_mix/scripts/build_mix.sh 2>&1 | tee /tmp/data_mix_smoke.log
```

Expected:
- Env table prints at top of log.
- HF requests fire for `liuhaotian/LLaVA-Instruct-150K` (or sibling)
  and `HuggingFaceTB/smol-smoltalk`.
- Produces `${DATA_MIX_OUTPUT_ROOT}/{train.jsonl, val_*.jsonl,
  build_report.json}`.

Production run (50K records, ~1-2 h depending on network):

```bash
unset CONFIG  # defaults to mix-50k.yaml
nohup bash src/data_mix/scripts/build_mix.sh \
  > ${DATA_MIX_OUTPUT_ROOT}/mix-50k.log 2>&1 &
echo "PID: $!"
disown
```

`nohup` is required — the LLaVA stream can run tens of minutes and an
SSH disconnect would otherwise kill the build (per
the long-running job guidance).

## Verify the mix loads through the finetune pipeline

```python
import sys
sys.path.insert(0, "src/finetune")
from src.data import load_vision_dataset
recs = load_vision_dataset(
    "${DATA_MIX_OUTPUT_ROOT}/train.jsonl",
    max_samples=100,
    require_image=False,   # v2 allows image=None for smoltalk / offline_qa
)
assert len(recs) == 100, "loader silently dropped records — schema problem"
```

A zero `n_dropped_no_image` count is mandatory when
`require_image=False`; the modality-aware sampler does the routing,
not a drop filter.

## Related

| File | Purpose |
|---|---|
| `src/data_mix/scripts/build_mix.sh` | The driver |
| `src/data_mix/src/mix.py` | Orchestrator entry point |
| `src/data_mix/src/env_paths.py` | Env-var resolver |
| `src/data_mix/configs/mix-*.yaml` | Per-mix configuration |
| [`02-bucket-design.md`](02-bucket-design.md) | What each bucket sampler produces |

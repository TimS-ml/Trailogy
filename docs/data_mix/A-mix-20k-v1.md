# A — mix-20k v1 (historical: Cambrian blocker → LLaVA pivot)

## TL;DR

- The first 20K-row mix combined plant images, general vision data, text-only chat, and non-plant refusal examples.
- The original general vision source failed during streaming, so the next version switched to a different image-instruction dataset.
- Text-only examples initially used a shared gray placeholder image because the loader expected image-bearing batches.
- A post-audit fix batch added missing tests and corrected determinism, image persistence, and plant-filtering issues.

## What v1 was meant to ship

A 20K-row mixed JSONL drop-in for
`finetune/src/data.py::load_vision_dataset`:

| Bucket | Share | Train | Val | Source |
|---|---:|---:|---:|---|
| Plant | 45 % | 9 000 | 450 | PlantNet enriched JSONL |
| **Cambrian** | 30 % | 6 000 | 300 | `nyu-visionx/Cambrian-10M` general subset |
| smoltalk | 15 % | 3 000 | 150 | `HuggingFaceTB/smol-smoltalk` — text-only, bound to shared dummy gray image |
| Negative | 10 % | 2 000 | 100 | non-plant Cambrian images + fixed refusal template |

Spec: `feature/quantization` branch on this repo, initial 13
commits landing 2026-05-15.

## What shipped — initial landing (13 commits)

```
b9bf956 feat(data_mix): mix-200 smoke-test config for HF/path validation
15311dd test(data_mix): end-to-end integration with stubbed HF streams
1158d27 feat(data_mix): build_mix.sh driver with env diagnostics + preflight
919622a feat(data_mix): mix-20k config (Plant 45 / Cambrian 30 / smoltalk 15 / Neg 10)
f697835 feat(data_mix): env-var resolver + 4-bucket orchestrator
9b53ba9 feat(data_mix): cambrian sampler with non-plant filter and resize
20045fa feat(data_mix): smoltalk text-only sampler bound to dummy image
2031c88 feat(data_mix): negative bucket builder (fixed refusal template)
0d85cc3 feat(data_mix): plant bucket sampler with per-class cap + 3 prompt variants
73d05e3 feat(data_mix): stretch-resize helper mirroring prepare_plantnet
ea0ae59 feat(data_mix): mid-gray 960x672 dummy image for smoltalk bucket
fea0c2d feat(data_mix): record schema validator with strict role alternation
c8ca852 feat(data_mix): scaffold module skeleton
```

40 / 40 unit + integration tests passed at this point (mocked HF
streams).

## Post-audit fix batch — 7 more commits

A code-review pass on 2026-05-15 night surfaced 5 real bugs + 4 test
gaps. Bug-2 was empirically verified non-triggering on the real
45K-row PlantNet data (0/45 000 problematic patterns) so it landed
as a docstring assumption rather than a code change. Gap-4 (the
`<image>` placeholder strip) was deferred because Cambrian was
already blocked.

```
07f7e1c test(data_mix): assert determinism — same seed produces byte-identical output
7f02009 test(data_mix): verify mix output is drop-in for finetune/src/data.py
0f051c3 refactor(data_mix): cambrian negative pool returns List[Path] not List[dict]
ddb836a fix(data_mix): word-boundary regex for plant filter rejects shared-prefix false positives
6afa339 fix(data_mix): _persist_image fast-path skip-if-exists + tmp cleanup on error
ebeab9b docs(data_mix): annotate plant_sampler first-sentence truncation assumption
e3fb96d fix(data_mix): replace assert with explicit InsufficientPoolError in _split_train_val
```

Bug taxonomy after the fix batch:

| ID | Severity | Fix |
|---|---|---|
| Bug-1 | Production blocker (silent under `-O`) | `InsufficientPoolError` |
| Bug-2 | Theoretical only (0 / 45 K real triggers) | Docstring assumption |
| Bug-3 | Performance + crash hygiene | Fast-path + try/finally |
| Bug-4 | False-positive filter rejections | Word-boundary regex |
| Bug-5 | Dead-metadata API smell | List[Path] for negative pool |
| Gap-1 | Spec contract uncovered | `finetune/src/data.py` compat test |
| Gap-2 | Determinism uncovered | Two-run byte-equality test |
| Gap-3 | `_persist_image` idempotency uncovered | Idempotency test |
| Gap-4 | `<image>` strip | Deferred to Cambrian-replacement task |

Test count: 40 → 54 after the fix batch.

## The blocker — Cambrian-10M streaming

`load_dataset("nyu-visionx/Cambrian-10M", streaming=True)` triggers a
`tarfile.ReadError: invalid header` on the very first record:

```
File "...datasets/packaged_modules/webdataset/webdataset.py", line 120
File "...tarfile.py", line 1765, in __init__
    self.firstmember = self.next()
File "...tarfile.py", line 2753, in next
    raise ReadError(str(e)) from None
tarfile.ReadError: invalid header
```

Root cause: the Cambrian-10M HF repo stores its image archives as
**multi-part `.tar.gz_partN` files** (e.g. `data_engine.tar.gz_part6`,
`allava.tar.gz_part12`) that must be concatenated and extracted
before they can be read. HF's `webdataset` streaming loader tries to
interpret each part as a self-contained tar archive and fails
immediately. The unit tests for `cambrian_sampler.py` all pass
because they inject a hand-crafted in-memory stream that mimics what
we *expected* HF to yield; the real HF stream never gets that far.

This is a Cambrian repo packaging issue, not a bug in our code.

## Three options considered

When the blocker surfaced, three independent unblock paths were on
the table:

**Option A: switch to a parquet-based VLM dataset.** Fastest. Use
`liuhaotian/LLaVA-Instruct-150K` or `HuggingFaceM4/the_cauldron` —
both parquet-backed and stream cleanly. Touch points: only
`cambrian_sampler.open_cambrian_stream()` needs to change. Adapt the
yielded record shape to the legacy `{"id", "image", "conversations"}`
shape so `sample_cambrian_records` (and its tests) stay untouched.
Estimate: 30 min code + 1–2 h for the actual stream.

**Option B: download Cambrian to disk and extract.** Keep Cambrian
as source of truth. Workflow: `huggingface-cli download` selected
non-multi-part tars (`coco.tar.gz`, `gqa.tar.gz`, `vg.tar.gz`) →
extract → rewrite `open_cambrian_stream` to iterate
`jsons/Cambrian7M.jsonl` from local disk. Estimate: ~30 min download
+ 30 min extract + 30 min code. Disk: ~50–150 GB.

**Option C: pin to an existing local image set** (e.g. COCO val on
some teammate's machine) and synthesize "What is in this image?" +
brief captions. Lowest-quality general-VLM diversity but unblocks
immediately.

## What was chosen — Option A (LLaVA)

Commit `f6d0c1f` (2026-05-16): `feat(data_mix): v2 — LLaVA-mix
source, image=None text-only, multi-val output`.

Three things changed simultaneously:

1. **Cambrian → LLaVA.** New `llava_sampler.py` replaced
   `cambrian_sampler.py` as the general-VLM source. The Cambrian
   sampler stays in the tree as reference but its config
   (`mix-20k`) does not run end-to-end.

2. **Dummy gray image → `image=None`.** The
   `ModalityAwareSFTTrainer` + `ModalityAwareBatchSampler` (commit
   `155cfc9` on the finetune side) made the dummy-image workaround
   unnecessary — text-only records are routed natively. smoltalk
   records now have `image=None`.

3. **Single val → multi val.** `val.jsonl` became
   `val_plant.jsonl` / `val_nonplant.jsonl` / `val_negative.jsonl`,
   threaded through `cfg.data.val_files` (dict) to the trainer for
   per-source `eval_<source>_loss` reporting.

## What happened to the v1 numbers

| Spec | Value at v1 close-out | Reality after v2 pivot |
|---|---:|---:|
| Bucket count | 4 | 4 (Plant / **LLaVA** / smoltalk / Negative) |
| Total train rows (target) | 20 000 | 20 000 → 50 000 (mix-50k canonical) → 100 000 (mix-100k) |
| Test count | 54 / 54 green | 96 / 96 green (incl. offline_qa, multi-val, drop-in test) |
| Real-network smoke | Blocked at step 0 | Runs end-to-end on `mix-200-llava.yaml` |

## What v2 changed (four things at once, commit `f6d0c1f`)

The v1 → v2 transition shipped as a single design landing. The four
simultaneous changes:

1. **Image source swap.** `nyu-visionx/Cambrian-10M` (blocked) →
   `HuggingFaceH4/llava-instruct-mix-vsft` (clean parquet stream).
   New `llava_sampler.py`; `cambrian_sampler.py` kept in the tree for
   reference.
2. **Two production sizes.** 100K + 50K instead of a single 20K.
   Plant per-class cap becomes the head/tail balance knob (50 in
   mix-50k for real balancing; 146 in mix-100k = full PlantNet train,
   no balancing).
3. **Mid-training multi-val + full ckpt retention.** Per-source val
   splits (`val_plant.jsonl` / `val_nonplant.jsonl` /
   `val_negative.jsonl`) wired into HF `SFTTrainer.eval_dataset` as a
   dict; `save_total_limit: null` so every checkpoint is kept for
   post-hoc best-of selection.
4. **smoltalk text-only skip-vision.** Drop the v1 dummy-image trick.
   smoltalk records get `image: None` and a new
   `ModalityAwareBatchSampler` + `ModalityAwareCollator` route
   text-only batches through a no-vision-tower forward path.
   Effective text token budget on smoltalk records: 744 → 1024
   (+ 37 %).

See [`B-mix-50k-v2.md`](B-mix-50k-v2.md) for the as-built details on
each of these four changes.

## Why v1 still has an entry in this doc set

Two reasons:

1. The 20K config is still in the tree
   (the historical `mix-20k` config) and someone reading the
   configs directory will hit it before any other doc. A pointer to
   "this is the v1 historical, use mix-50k" prevents wasted attempts.
2. The Cambrian module is still in the tree
   (`data_mix/src/cambrian_sampler.py`) for the same reason. If the
   Cambrian repo ever fixes their tar packaging the sampler becomes
   useful again without a re-write.

## Cross-refs

- [`02-bucket-design.md`](02-bucket-design.md) — v2 schema, including
  `image=None` and the modality-aware sampler change.
- [`B-mix-50k-v2.md`](B-mix-50k-v2.md) — the current production mix.

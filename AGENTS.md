# AGENTS.md — Trailogy

Context checkpoint for future agent sessions (Codex, Claude Code, etc.).
Read this first. `CLAUDE.md` is a symlink to this file — keep edits
here.

## What this is

iOS app for the **Kaggle Gemma 4 for Good** hackathon. The product is a
"hike companion" that explains what hikers see in front of them and
answers grounded questions about a specific trail — geology, plants,
ecology, local history. **Everything runs on-device**: no network at
runtime, no cloud inference, no streaming.

Prototype is end-to-end complete. Five integrated subsystems:

- **Gemma 4 E2B** (INT4, ~2.8 GB on disk, ~2.5–2.78 GB MLX active) for
  the LLM — multimodal `mlx-community/gemma-4-e2b-it-4bit` checkpoint,
  audio tower stripped via `scripts/strip-gemma-audio.py`. Dual-mode
  loader: `.text` (MLXLLM) for typed/voice Q&A, `.vlm` (MLXVLM) for
  image Q&A.
- **MiniLM (swift-embeddings)** for RAG — ~87 MB FP16, bundled, drives
  cosine-similarity search over per-trail subject corpora in `rag-poc/`.
- **Kokoro 82M** (FP32 safetensors, ~327 MB) for TTS.
- **SFSpeechRecognizer** (Apple, on-device) for voice input.
- **AVCaptureSession** for camera capture.

User flows:

- **Voice text Q&A**: hold mic → speak → release → Gemma `.text`
  (with RAG context) → Kokoro speaks.
- **Image Q&A**: tap camera → capture → photo-context strip appears →
  hold mic → ask about photo → Gemma `.vlm` (`[camera=on]` prefix) →
  Kokoro speaks. Photo cleared after a successful answer.
- **Trail recap**: at trail end, Gemma summarizes per-hike takeaways.

Conversation memory persists in `GemmaService` across unload/reload.
Cap: 20 messages (10 turns) for text Asks; 0 history for image Asks
(image already prefills ~280 vision tokens).

## Repo

- GitHub: `git@github.com:YingCeci/Trailogy.git` (collaborator's
  account; secondary `TimS-ml/Trailogy` mirror also force-pushes here).
  Repo was renamed from `hikeCompanion` to `Trailogy` mid-development;
  Xcode target name keeps the legacy `HikeCompanion` to avoid churning
  bundle ID + provisioning (user-facing display name is "Trailogy").
- Local clone directory not renamed from `hikeCompanion`; cosmetic only.
- Owner intent: Billy Li (`lijuncheng16`); collaborator: Ying Wang
  (`YingCeci`). Bundle ID is `com.lijuncheng16.HikeCompanion`.

## Repo conventions — non-negotiable rules

These rules are enforced for the public repo and have been retro-active
on git history (filter-repo scrub on 2026-05-23):

1. **No absolute paths in tracked files.** Image roots, python prefixes,
   `/home/<user>/...`, `/Users/<user>/...`, conda paths — none belong
   in committed configs, data, scripts, or docs. Use env vars
   (`$PLANT_IMAGE_ROOT`, `$DATA_ROOT`, `$HF_HOME`), CLI flags, or
   relative paths joined at runtime.
2. **No specific hardware names.** Generic GPU names (`4090`, `A100`,
   `L40S`) are fine; "Tim's 4090 laptop" or "4x4090 box" is not.
3. **mix-50k-v2 is the default training corpus** for every active
   config under `src/finetune/configs/{cloud_sweep,local_sweep}/`. Old
   v1 baselines live in `src/finetune/configs/archive/` and are
   historical record only.
4. **bf16 LoRA SFT only by default.** 4-bit / 8-bit anything (4-bit
   training, bitsandbytes optimizers, fp8 Adam) is forbidden unless the
   work item is explicitly labelled as quantization-side. This is to
   keep SFT bake-offs comparable.
5. **Backup files (`*.jsonl.pre_*`, `*.bk`, `*.bak`)** are gitignored.
   They tend to leak host-specific abs paths and shouldn't be
   committed. The `.gitignore` enforces this defensively.
6. **No references to private sibling repos** by name in tracked
   files. They exist; we don't quote them in the public tree.

## Tech stack — what's vendored and why

Four packages live under `external/` as **vendored source copies**:

- `external/kokoro-ios` — KokoroSwift 1.0.11 with `mlx-swift` pin
  relaxed.
- `external/MisakiSwift` — same reason, sibling `path:` dep.
- `external/MLXUtilsLibrary` — same reason; also re-adds a no-op
  `BenchmarkTimer` stub (KokoroSwift 1.0.11 calls it, but it was
  removed in MLXUtilsLibrary 0.0.7+). **Don't delete that stub file.**
- `external/mlx-swift-lm` — vendored so we can patch
  `Libraries/MLXVLM/Models/Gemma4.swift` for the VLM image pipeline.

**Do not replace any of these with URL-based deps** without
understanding the conflicts:

- **mlalma's KokoroSwift 1.0.11** hard-pins `mlx-swift exact:
  "0.30.2"`. So do MisakiSwift 1.0.6 and MLXUtilsLibrary 0.0.6.
- **mlx-swift-lm 3.x** (the only version with Gemma 4 support) requires
  `mlx-swift 0.31+`. Direct conflict — fixed by relaxing the Kokoro
  packages' MLX pins to ranges in their vendored `Package.swift`.

URL-based SPM deps (in `project.yml`):

- `swift-embeddings` ≥ 0.0.10 — MiniLM loader for RAG (downloads on
  first use, caches in app sandbox; bundled copy under
  `Resources/Models/MiniLM/` planned).
- `swift-transformers` ≥ 1.3.0 — products `Tokenizers`, `Hub`.
- `swift-huggingface` ≥ 0.8.1 — product `HuggingFace` (needed for the
  `#huggingFaceTokenizerLoader()` macro to compile).

Macros require **explicit trust on first Xcode open** ("Trust & Enable
All"). For CLI builds, pass `-skipMacroValidation`.

## Critical lifecycle patterns — DO NOT REGRESS

### Gemma is lazy-loaded per Ask, unloaded after generation, dual-mode

`GemmaService.loadIfNeeded(_ kind: LoadedKind)` is called at the start
of every Ask with `.text` or `.vlm`; `gemma.unload()` runs after
generation completes. If a different `kind` is currently loaded,
`loadIfNeeded` unloads first. This:

- Pays a 10–30 s reload per Ask (model file mmap + MLX kernel JIT).
  VLM mode is ~3–5 s slower (vision-tower kernels add to the JIT
  pass).
- **Bounds memory.** Keeping Gemma resident across the Gemma → Kokoro
  hand-off OOM'd the app even on iPhone 17 Pro.
- **Conversation history persists in `GemmaService` itself**, not in
  the ModelContainer — survives unload/reload. Replayed into a fresh
  `ChatSession` per call.
  - Text asks: `maxHistoryMessages = 20` (10 turns).
  - Image asks: `maxImageHistoryMessages = 0` — image already prefills
    ~280 vision tokens; replaying chat history on top inflates KV
    cache near the jetsam line.

### Kokoro uses a TWO-PHASE serial workQueue unload

In `ValidationRunner.synthesize`:

```swift
workQueue.async { /* phase 1: synth + play. Local `tts` binding alive. */ }
workQueue.async { /* phase 2: self.tts = nil; Memory.clearCache() */ }
```

**Why two phases on a serial queue**: phase 1 captures `let tts =
self.tts` locally. That binding lives until phase 1 closure exits. If
we set `self.tts = nil` and `Memory.clearCache()` from main during
phase 1's execution, the cache clears *before* the local binding is
released, and when ARC eventually frees the model the buffers go right
back into MLX's cache pool — which we never clear again. Phase 2 on
the same serial queue runs only after phase 1 fully exits, so the
local ref is gone by then. **Don't merge these into one async block.**

### MLX Memory cap is removed, but cache is cleared between Gemma and Kokoro

`HikeCompanionApp.init()` sets `Memory.cacheLimit = 100 MB` only — no
`Memory.memoryLimit`. A hard ceiling forced MLX to allocate at the
critical path during the Gemma → Kokoro hand-off and tripped jetsam.
Without a cap, MLX sizes its own arena steadier.

`MLX.Memory.clearCache()` is called at the end of `GemmaService.unload()`
to drop transient buffers before Kokoro starts.

### RAG embedder stays resident; retrieved context is one-shot

`RAGService` keeps the MiniLM embedder loaded (~87 MB FP16) across
Asks because it is small next to Gemma. The retrieved top-k chunks are
inserted into Gemma's prompt for that **one** answer and cleared on
the next Ask. The embedder evaluates on the Neural Engine when
available.

Per-trail subject activation lives on the trail definition (e.g.
"geology + plants" vs "wetlands + history"); `DebugView` overrides the
active set at runtime for testing.

### iOS jetsam entitlement — required for VLM

`HikeCompanion/HikeCompanion.entitlements` carries:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
```

Default iPhone Pro foreground jetsam is ~3.5 GB process footprint.
**VLM peak is ~3.5 GB** (vision encoder fixed at 2520 tokens, language
prefill adds 250–400 MB transient). Without this entitlement, the app
silently jetsam-kills mid-prefill on the first image Ask. The
entitlement raises the ceiling to ~6 GB and is available on both free
and paid Apple Developer accounts.

## Bundle layout (xcodegen)

```yaml
sources:
  - path: HikeCompanion
    excludes:
      - "Resources/Models/**"
  - path: HikeCompanion/Resources/Models
    type: folder      # NO `buildPhase: resources` — that flattens contents
```

`type: folder` (without `buildPhase: resources`) creates a **blue-folder
reference** that preserves the directory tree:

- `HikeCompanion.app/Models/kokoro-v1_0.safetensors`
- `HikeCompanion.app/Models/voices.npz`
- `HikeCompanion.app/Models/Gemma/config.json` + `model.safetensors` etc.
- `HikeCompanion.app/Models/MiniLM/...` (when pre-bundled — currently
  downloaded on first launch).
- `HikeCompanion.app/Resources/RAG/*.jsonl` + `.embeddings.f16` per
  subject.

**This separation is critical**: mlx-swift-lm globs `*.safetensors` in
the directory you hand it. If Kokoro's safetensors and Gemma's were
both at the bundle root, the loader would try to load Kokoro's BERT
weights into the Gemma4Model graph and crash with `"Unhandled keys
[bert, decoder, …]"`.

In Swift, look up bundle resources with `subdirectory: "Models"`:

```swift
Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors",
                subdirectory: "Models")
```

## Memory profile (iPhone 17 Pro, 12 GB RAM)

Both Gemma kinds unload between turns (MLX active returns to 14 MB).

### Text-only Ask

| State | Process | MLX active | MLX peak |
|---|---|---|---|
| Cold start | 41 MB | 14 MB | 14 MB |
| Idle between Asks | ~100 MB | 14 MB | (lifetime) |
| Gemma `.text` loaded | ~2.6 GB | 2.47 GB | 2.55 GB |
| Generation | ~2.8 GB | 2.47 GB | 2.55 GB |
| After Gemma unload | ~150 MB | **14 MB** | 2.55 GB |
| Kokoro speaking | ~600 MB | ~324 MB | 2.55 GB |

### Image Ask (VLM mode)

| State | Process | MLX active | MLX peak |
|---|---|---|---|
| Camera capture | ~170 MB | 14 MB | (carries over) |
| Gemma `.vlm` loaded | ~3.0 GB | 2.78 GB | 2.78 GB |
| Vision tower forward | ~3.0 GB | ~2.87 GB | 3.22 GB |
| Language prefill eval | ~3.1 GB | ~2.79 GB | **~3.5 GB** |
| After Gemma unload | ~360 MB | **14 MB** | 3.5 GB |

iOS default foreground jetsam ~3.5 GB on iPhone Pro models — the
`increased-memory-limit` entitlement raises this to ~6 GB. Required
for VLM.

## Project layout

```
Trailogy/
├── README.md, AGENTS.md, CLAUDE.md (symlink → AGENTS.md),
│   project.yml, .gitignore
├── design/                         # source HTML mockups
├── docs/                           # engineering deep dives
│   ├── general/                    # architecture, runtime, eval, postmortems
│   ├── data_mix/                   # mix corpus design + build notes
│   ├── finetune/                   # SFT pipeline + anti-forgetting recipe
│   └── quantization/               # deploy-time compression + results
├── scripts/                        # iOS build helpers
│   ├── fetch-models.sh             # Kokoro safetensors + voices.npz
│   ├── fetch-gemma.sh              # Gemma 4 E2B + processor_config patch
│   ├── strip-gemma-audio.py        # drop ~580 MB audio_tower weights
│   └── generate-project.sh         # xcodegen wrapper
├── external/                       # vendored SPM packages (patched)
│   ├── kokoro-ios/                 # MLX pin relaxed
│   ├── MisakiSwift/                # MLX pin relaxed, sibling path: dep
│   ├── MLXUtilsLibrary/            # BenchmarkTimer stub re-added
│   └── mlx-swift-lm/               # Gemma4.swift patched for VLM
├── rag-poc/                        # bundled RAG corpora (.jsonl per subject)
├── src/                            # Python — training + data + quant
│   ├── data_mix/                   # mixed SFT corpus builder
│   ├── finetune/                   # LoRA SFT pipeline (unsloth)
│   └── quantization/               # deploy-time quantization sweep
└── HikeCompanion/                  # iOS app (Xcode target name preserved)
    ├── HikeCompanionApp.swift      # @main; MLX cache limit + memory ticker
    ├── ContentView.swift           # router root
    ├── AppRouter.swift             # @Published screen FSM
    ├── Theme.swift, TrailData.swift, MemoryStats.swift
    ├── GemmaService.swift          # dual-mode loader (.text/.vlm)
    ├── RAGService.swift            # MiniLM + cosine search over rag-poc/
    ├── ValidationRunner.swift      # Kokoro wrapper, two-phase unload
    ├── SpeechRecognizer.swift      # SFSpeechRecognizer wrapper
    ├── CameraController.swift      # AVCaptureSession + AVCapturePhotoOutput
    ├── ImageStore.swift            # offline on-disk image cache
    ├── HikeCompanion.entitlements  # increased-memory-limit
    ├── Views/                      # SwiftUI screens
    │   ├── PickerView.swift, DetailView.swift, WalkingView.swift
    │   ├── JournalView.swift, RecapView.swift
    │   ├── CameraView.swift, CameraPreviewView.swift
    │   ├── TourMapView.swift, TrailMapShape.swift, DebugView.swift
    ├── Info.plist (generated)
    ├── Assets.xcassets/
    └── Resources/Models/           # gitignored — fetch via scripts above
```

## Python pipeline (`src/`)

The Python tree drives the model that gets bundled into the iOS app.
Three workstreams, each with its own README + tests + configs:

- **`src/data_mix/`** — builds the mixed SFT corpus. Combines a
  PlantNet-derived plant slice with LLaVA/SmolTalk/refusal buckets to
  prevent catastrophic forgetting. The active default corpus is
  `mix-50k-v2` (~50 k rows, na_plantae 60 % + general 40 %).
  Per-record `[camera=on]` / `[camera=off]` prompt-prefix tags gate
  the iOS runtime modality. See
  [`docs/data_mix/B-mix-50k-v2.md`](docs/data_mix/B-mix-50k-v2.md).

- **`src/finetune/`** — LoRA SFT of Gemma 4 E2B with unsloth on a
  single NVIDIA GPU. Sweeps live under
  `configs/{cloud_sweep,local_sweep}/`. Active rank grid: r=8/16/32/64/256.
  Driver scripts under `scripts/run/` (`train.sh`,
  `local_sweep_r-kl-vision.sh`). Eval harness in `eval/`
  (`evaluate_generality.py` runs plant/mmlu/aime/llava/refusal). See
  [`docs/finetune/03-anti-forgetting-and-final-recipe.md`](docs/finetune/03-anti-forgetting-and-final-recipe.md)
  for the shipped recipe and
  [`docs/finetune/10-no-text-prefix-and-bigger-rank.md`](docs/finetune/10-no-text-prefix-and-bigger-rank.md)
  for the bigger-rank scan.

- **`src/quantization/`** — deploy-time compression of the bf16-merged
  adapter down to the ~3.6 GB iOS-loadable artifact, with EoRA
  recovery. See
  [`docs/quantization/00-quantization-report-pub.md`](docs/quantization/00-quantization-report-pub.md).

### Shipped model

The bundled Gemma is the result of:

1. **Stage-1 SFT** (`r8-a8-nokl`) — small-rank LoRA on `mix-50k-v2`
   with `[camera=on/off]` prefix gating. Anti-forgetting via data
   mixing + modality tags; KL/L2 documented as fallback controls.
2. **Stage-2 SFT** — short NA-Plantae tree-specialized adapter on top
   of stage-1, closing the gap on common North-American trees that
   PlantNet under-represents (see
   [`docs/general/16-final-model-eval.md`](docs/general/16-final-model-eval.md)).
3. **MLX 4-bit quantization** via `src/quantization/` then bundled into
   `HikeCompanion/Resources/Models/Gemma/`.

### Finetune → MLX export contract — DO NOT REGRESS

`src/finetune/src/export_mlx.py` enforces three vision-preservation
invariants that iOS depends on. These are guarded by tripwires and
unit tests; if any break, export should fail loudly rather than ship a
model that quietly drops the vision tower:

1. **Merge uses `AutoModelForImageTextToText`**, NOT
   `AutoModelForCausalLM`. The CausalLM auto-class only loads the
   language sub-module and silently drops `vision_tower.*` /
   `embed_vision.*`.
2. **MLX conversion uses `mlx_vlm.convert`**, NOT `mlx_lm.convert`.
   `mlx_lm` is language-only and silently drops vision weights at
   sanitize time.
3. **`processor_config.json` is patched to `size: {height: 960, width:
   672}`** after conversion. This matches the trained shape and what
   `scripts/fetch-gemma.sh` patches on the raw checkpoint, so the iOS
   bundle gets the trained shape regardless of fetch order.

The shape constant `TRAINED_VISION_SIZE = {"height": 960, "width":
672}` is defined in `src/finetune/src/export_mlx.py` and
`DEFAULT_TRAINED_VISION_HW` in `src/finetune/src/prepare_plantnet.py`.
**Both must match `scripts/fetch-gemma.sh`'s `TRAINED_SIZE`.**

## Phase status

- ✅ **Phase 1** — typed text → Gemma → Kokoro, multi-turn memory.
- ✅ **Phase 2** — voice input via SFSpeechRecognizer (hold-to-speak).
- ✅ **Phase 3a** — camera capture via AVCaptureSession.
- ✅ **Phase 3b** — image Q&A via MLXVLM Gemma 4 multimodal, with
  the `increased-memory-limit` entitlement.
- ✅ **Phase 4** — RAG over per-trail subject corpora via MiniLM
  embedder + cosine search.
- ✅ **Phase 5** — trail recap (Gemma-generated takeaways at trail end),
  offline image cache, modality-aware prompt gating.

## Setup commands (cold clone)

```bash
git clone --recurse-submodules git@github.com:YingCeci/Trailogy.git
# (no actual submodules — `external/` is committed directly — but the
# recurse flag is harmless if we ever switch back)
cd hikeCompanion       # local clone dir not renamed

bash scripts/fetch-models.sh           # Kokoro: ~630 MB
bash scripts/fetch-gemma.sh            # Gemma 4 E2B: ~3.4 GB (add --backup for unsloth fallback)
python3 scripts/strip-gemma-audio.py   # Optional: strips ~580 MB audio_tower
bash scripts/generate-project.sh

open HikeCompanion.xcodeproj
# In Xcode: trust macros when prompted; set Development Team in Signing & Capabilities
# ⌘R to a real iPhone (≥ iPhone 15 Pro / iOS 18). Simulator does not have MLX.
```

### Audio-tower strip (why ~2.8 GB instead of ~3.4 GB)

The HF checkpoint is the **multimodal** Gemma 4 E2B — it carries
language_model + vision_tower + audio_tower. mlx-swift-lm filters
audio weights at sanitize() time (in both MLXLLM and MLXVLM Gemma 4
loaders), so they're never used by the iPhone runtime.
`scripts/strip-gemma-audio.py` reads `model.safetensors` and writes a
new file without the 754 `audio_tower.*` / `embed_audio.*` tensors —
saves ~583 MB on disk with zero functional impact.

The script keeps a `.audio.bak` copy as a safety net at
`scripts/backups/model.safetensors.audio.bak` — **outside** the bundle
resource path so Xcode's `type: folder` reference for
`Resources/Models` doesn't sweep it into the `.app`. (Critical: the
very first run of the strip script put the backup *inside*
`Resources/Models/Gemma/`, which bloated the `.app` bundle from ~3.1
GB to ~6.4 GB. The script now defaults to `scripts/backups/` and
migrates any legacy backup it finds.)

To restore:
```
mv scripts/backups/model.safetensors.audio.bak \
   HikeCompanion/Resources/Models/Gemma/model.safetensors
```
(or just re-run `bash scripts/fetch-gemma.sh` to pull a fresh copy.)

## Known gotchas

### iOS

- **Build for iOS Simulator** works for compile verification but **the
  app cannot run on Simulator** — MLX requires Metal compute that the
  simulator doesn't have.
- **iPhone needs Developer Mode enabled** (Settings → Privacy &
  Security → Developer Mode → on, then reboot) before any sideloaded
  build can launch.
- **Free Apple ID dev certs expire after 7 days** — re-run from Xcode
  each week if not on a paid Developer Program account.
- **Release build doesn't work** out of the box — Xcode 26's strict
  module scanner fails on transitive deps (`Atomics`, `DequeModule`,
  `Numerics`). Use Debug. To reduce Debug overhead: scheme → Run →
  Diagnostics → uncheck Main Thread Checker and Thread Performance
  Checker.
- **TTS on long input glitches** if not chunked — KokoroSwift's
  duration predictor goes unstable past ~60 chars. `splitForSynthesis`
  in `ValidationRunner` splits on `. ! ? , : ;` with `maxCharsPerChunk
  = 80`. **Don't raise above 80**; below ~60 chars the model behaves.
- **VLM Q&A requires the `increased-memory-limit` entitlement.**
  Without it, iPhone foreground jetsam (~3.5 GB) is below VLM peak
  (~3.5 GB), and the app silently dies during prefill eval.
- **VLM forces 960×672 portrait shape.** Aspect-ratio-preserving
  resize is not yet ported in `mlx-swift-lm`; square and landscape
  photos get stretched. Matches the trained pooler so recognition is
  acceptable but not great. See
  [`docs/general/13-mlx-vision-input-parity.md`](docs/general/13-mlx-vision-input-parity.md).

### Python pipeline

- **Some package versions ship with bugs we work around** — see
  [`docs/general/14-package-versions-and-known-bugs.md`](docs/general/14-package-versions-and-known-bugs.md).
  Notable: `transformers` Gemma 4 attention implementation, `peft`
  silently dropping `modules_to_save` weights on reload, `unsloth`
  pinning buggy model revisions and treating "4-bit" as a 4–8-bit mix.
- **Terminal timeouts during long sweeps.** Each LoRA run can be hours.
  Wrap with `nohup` / `tmux` / the supplied sweep driver
  (`scripts/run/local_sweep_*.sh`).
- **Wandb defaults to online.** Set `WANDB_MODE=offline` on
  air-gapped boxes and sync later. The sweep driver does this
  automatically.

## Where to learn more — `docs/`

Engineering deep dives live under `docs/`. Read [`docs/README.md`](docs/README.md)
for the full reading order. Shortest path to the technical story:

| Order | Read | Why it matters |
|---:|---|---|
| 1 | [`docs/general/02-architecture-ios-app.md`](docs/general/02-architecture-ios-app.md) | Offline product, runtime stack, why memory discipline matters. |
| 2 | [`docs/general/01-architecture-model-pipeline.md`](docs/general/01-architecture-model-pipeline.md) | How data, SFT, quantization, and iOS deployment connect. |
| 3 | [`docs/finetune/03-anti-forgetting-and-final-recipe.md`](docs/finetune/03-anti-forgetting-and-final-recipe.md) | Catastrophic-forgetting problem and the recipe that shipped. |
| 4 | [`docs/data_mix/B-mix-50k-v2.md`](docs/data_mix/B-mix-50k-v2.md) | The mixed corpus that prevented plant-only collapse. |
| 5 | [`docs/quantization/00-quantization-report-pub.md`](docs/quantization/00-quantization-report-pub.md) | How the 9.5 GB bf16 model became iOS-loadable. |
| 6 | [`docs/general/15-postmortems.md`](docs/general/15-postmortems.md) | Silent failures that could have invalidated the results. |

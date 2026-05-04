# CLAUDE.md — hikeCompanion

Context checkpoint for future Claude sessions. Read this first.

## What this is

iOS app for the **Kaggle Gemma 4 for Good** hackathon. The product idea is a
"hike companion" that explains what hikers see in front of them — geology,
plants, climate change in national parks. Everything runs **on-device**:

- **Gemma 4 E2B** (INT4) for the LLM, dual-mode loader:
  - `.text` mode: ~2.5 GB MLX active (text decoder only via `MLXLLM`)
  - `.vlm` mode: ~2.78 GB MLX active (text decoder + vision tower via `MLXVLM`)
  - Checkpoint is the multimodal `mlx-community/gemma-4-e2b-it-4bit`
    (~2.8 GB on disk after audio-tower strip).
- **Kokoro 82M** (FP32 safetensors, ~327 MB) for TTS
- **SFSpeechRecognizer** (Apple, on-device) for voice input
- **AVCaptureSession** for camera capture (Phase 3a)

User flows:
- **Text Q&A**: hold mic → speak question → release → Gemma (text) streams
  response → Kokoro speaks it aloud. Multi-turn memory across 10 turns.
- **Image Q&A**: tap camera → frame + capture → photo-context strip
  appears → hold mic → ask about photo → Gemma (VLM) streams response →
  Kokoro speaks it aloud. Photo cleared after a successful answer.

`GemmaService.LoadedKind` (`.text` / `.vlm`) is decided per turn based
on whether `WalkingView.capturedImage` is non-nil; the service unloads
the wrong-kind model and loads the right one before streaming.

## Repo

- GitHub: `git@github.com:YingCeci/hikeCompanion.git` (private — under
  YingCeci, not lijuncheng16, due to the SSH key on this Mac)
- Local path: ``
- Owner intent: Billy Li (`lijuncheng16`); collaborator: Ying Wang (`YingCeci`).
  Bundle ID is `com.lijuncheng16.HikeCompanion`.

## Tech stack — what's vendored and why

Four packages live under `external/` as **vendored source copies**:

- `external/kokoro-ios` — KokoroSwift 1.0.11 with `mlx-swift` pin relaxed
- `external/MisakiSwift` — same reason, sibling `path:` dep
- `external/MLXUtilsLibrary` — same reason; also re-adds a no-op
  `BenchmarkTimer` stub (KokoroSwift 1.0.11 calls it, but it was removed
  in MLXUtilsLibrary 0.0.7+). **Don't delete that stub file.**
- `external/mlx-swift-lm` — vendored so we can patch
  `Libraries/MLXVLM/Models/Gemma4.swift` (see "Phase 3b VLM patches" below).

**Do not replace any of these with URL-based deps** without understanding
the conflicts:

- **mlalma's KokoroSwift 1.0.11** hard-pins `mlx-swift exact: "0.30.2"`.
  So do MisakiSwift 1.0.6 and MLXUtilsLibrary 0.0.6.
- **mlx-swift-lm 3.x** (the only version with Gemma 4 support) requires
  `mlx-swift 0.31+`. Direct conflict — fixed by relaxing the Kokoro
  packages' MLX pins to ranges in their vendored `Package.swift`.

URL-based SPM deps (in `project.yml`):
- `swift-transformers` ≥ 1.3.0 — products `Tokenizers`, `Hub`
- `swift-huggingface` ≥ 0.8.1 — product `HuggingFace` (needed for the
  `#huggingFaceTokenizerLoader()` macro to compile)

Macros require **explicit trust on first Xcode open** ("Trust & Enable All").
For CLI builds, pass `-skipMacroValidation`.

## Critical lifecycle patterns — DO NOT REGRESS

### Gemma is lazy-loaded per Ask, unloaded after generation, dual-mode

`GemmaService.loadIfNeeded(_ kind: LoadedKind)` is called at the start of
every Ask with `.text` or `.vlm`; `gemma.unload()` runs after generation
completes. If a different `kind` is currently loaded, `loadIfNeeded`
unloads first. This:

- Pays a 10–30 s reload per Ask (model file mmap + MLX kernel JIT).
  VLM mode is ~3–5 s slower (vision-tower kernels add to the JIT pass).
- **Bounds memory**. Keeping Gemma resident across the Gemma → Kokoro
  hand-off OOM'd the app even on iPhone 17 Pro.
- **Conversation history persists in `GemmaService` itself**, not in the
  ModelContainer — survives unload/reload. Replayed into a fresh
  `ChatSession` per call.
  - Text asks: `maxHistoryMessages = 20` (10 turns).
  - Image asks: `maxImageHistoryMessages = 0` — image already prefills
    ~280 vision tokens; replaying chat history on top inflates KV cache
    near the jetsam line.

### Kokoro uses a TWO-PHASE serial workQueue unload

In `ValidationRunner.synthesize`:

```swift
workQueue.async { /* phase 1: synth + play. Local `tts` binding alive. */ }
workQueue.async { /* phase 2: self.tts = nil; Memory.clearCache() */ }
```

**Why two phases on a serial queue**: phase 1 captures `let tts = self.tts`
locally. That binding lives until phase 1 closure exits. If we set
`self.tts = nil` and `Memory.clearCache()` from main during phase 1's
execution, the cache clears *before* the local binding is released, and
when ARC eventually frees the model the buffers go right back into MLX's
cache pool — which we never clear again. Phase 2 on the same serial
queue runs only after phase 1 fully exits, so the local ref is gone by
then. **Don't merge these into one async block.**

### MLX Memory cap is removed, but cache is cleared between Gemma and Kokoro

`HikeCompanionApp.init()` sets `Memory.cacheLimit = 100 MB` only — no
`Memory.memoryLimit`. A hard ceiling forced MLX to allocate at the critical
path during the Gemma → Kokoro hand-off and tripped jetsam. Without a cap,
MLX sizes its own arena steadier.

`MLX.Memory.clearCache()` is called at the end of `GemmaService.unload()`
to drop transient buffers before Kokoro starts.

### iOS jetsam entitlement — required for VLM

`HikeCompanion/HikeCompanion.entitlements` contains:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
```

Default iPhone Pro foreground jetsam is ~3.5 GB process footprint.
**VLM peak is ~3.54 GB** (vision encoder is fixed at 2520 tokens,
language prefill adds ~250-400 MB transient). Without this entitlement,
the app silently jetsam-kills mid-prefill on the first image Ask. The
entitlement raises the ceiling to ~6 GB.

Available on both free and paid Apple Developer accounts (xcodegen
auto-generates the file if missing). If signing rejects this entitlement,
delete the `entitlements:` block from `project.yml` — but image Q&A
will fail.

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

**This separation is critical**: mlx-swift-lm globs `*.safetensors` in the
directory you hand it. If Kokoro's safetensors and Gemma's were both at
the bundle root, the loader would try to load Kokoro's BERT weights into
the Gemma4Model graph and crash with `"Unhandled keys [bert, decoder, …]"`.

In Swift, look up bundle resources with `subdirectory: "Models"`:
```swift
Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors",
                subdirectory: "Models")
```

## Memory profile (iPhone 17 Pro, 12 GB RAM)

After all the optimizations:

### Text-only Ask

| State | Process footprint | MLX active | MLX peak |
|---|---|---|---|
| Cold start | 41 MB | 14 MB | 14 MB |
| Idle between Asks | ~100 MB | 14 MB | (lifetime) |
| Gemma `.text` loaded | ~2.6 GB | 2.47 GB | 2.55 GB |
| Generation | ~2.8 GB | 2.47 GB | 2.55 GB |
| After Gemma unload | ~150 MB | **14 MB** | 2.55 GB |
| Kokoro speaking | ~600 MB | ~324 MB | 2.55 GB |

### Image Ask (VLM mode)

| State | Process footprint | MLX active | MLX peak |
|---|---|---|---|
| Camera capture | ~170 MB | 14 MB | (carries over) |
| Gemma `.vlm` loaded | ~3.0 GB | 2.78 GB | 2.78 GB |
| Vision tower forward | ~3.0 GB | ~2.87 GB | 3.22 GB |
| Language prefill (lazy graph built) | ~3.0 GB | 2.85 GB | 3.22 GB |
| Prefill eval + decode | ~3.1 GB | ~2.79 GB | **~3.54 GB** |
| After Gemma unload | ~360 MB | **14 MB** | 3.54 GB |

iOS default foreground jetsam ~3.5 GB on iPhone Pro models.
**`increased-memory-limit` entitlement raises this to ~6 GB** —
required for VLM. Both models still successfully unload between
turns (MLX active = 14 MB).

## Project layout

```
hikeCompanion/
├── README.md, CLAUDE.md, AGENTS.md, ARCHITECTURE.md, project.yml, .gitignore
├── design/
│   └── mockups.html               # source of the SwiftUI redesign
├── scripts/
│   ├── fetch-models.sh            # Kokoro safetensors + voices.npz
│   ├── fetch-gemma.sh             # Gemma 4 E2B + processor_config patch (hoist + 960×672)
│   ├── strip-gemma-audio.py       # Remove ~580 MB audio_tower weights from safetensors
│   ├── backups/                   # gitignored — strip-gemma-audio.py output lives here
│   └── generate-project.sh        # xcodegen wrapper
├── external/
│   ├── kokoro-ios/                # vendored, MLX pin relaxed
│   ├── MisakiSwift/               # vendored, MLX pin relaxed, sibling path: dep
│   ├── MLXUtilsLibrary/           # vendored, BenchmarkTimer stub re-added
│   └── mlx-swift-lm/              # vendored — Gemma4.swift has [Gemma4.mem] instrumentation
└── HikeCompanion/
    ├── HikeCompanionApp.swift     # @main; MLX cache limit + MemoryProbe ticker
    ├── ContentView.swift          # Router root: switches picker/detail/walking/journal
    ├── AppRouter.swift            # @Published screen state machine
    ├── Theme.swift, TrailData.swift, MemoryStats.swift
    ├── GemmaService.swift         # dual-mode loader (.text/.vlm), history replay
    ├── ValidationRunner.swift     # Kokoro wrapper, two-phase unload
    ├── SpeechRecognizer.swift     # SFSpeechRecognizer wrapper
    ├── CameraController.swift     # AVCaptureSession + AVCapturePhotoOutput
    ├── HikeCompanion.entitlements # increased-memory-limit (auto-generated by xcodegen)
    ├── Views/
    │   ├── PickerView.swift, DetailView.swift, WalkingView.swift, JournalView.swift
    │   ├── CameraView.swift, CameraPreviewView.swift, TourMapView.swift
    │   ├── TrailMapShape.swift, DebugView.swift
    ├── Info.plist (generated)
    ├── Assets.xcassets/           # AppIcon, AccentColor, LaunchBg
    └── Resources/Models/          # gitignored — fetch via scripts above
```

See `ARCHITECTURE.md` for a one-page diagram of the layers + Q&A flow.

## Setup commands (cold clone)

```bash
git clone --recurse-submodules git@github.com:YingCeci/hikeCompanion.git
# (no actual submodules — `external/` is committed directly — but keeping
# the recurse flag harmless if we ever switch back)
cd hikeCompanion

bash scripts/fetch-models.sh           # Kokoro: ~630 MB
bash scripts/fetch-gemma.sh            # Gemma 4 E2B: ~3.4 GB. Add --backup for unsloth fallback.
python3 scripts/strip-gemma-audio.py   # Optional: strips ~580 MB of unused audio-tower weights
bash scripts/generate-project.sh

open HikeCompanion.xcodeproj
# In Xcode: trust macros when prompted; set Development Team in Signing & Capabilities
# ⌘R to a real iPhone (≥ iPhone 15 Pro / iOS 18). Simulator does not have MLX.
```

### Audio-tower strip (why ~2.8 GB instead of ~3.4 GB)

The HF checkpoint is the **multimodal** Gemma 4 E2B — it carries
language_model + vision_tower + audio_tower. mlx-swift-lm filters audio
weights at sanitize() time (in both MLXLLM and MLXVLM Gemma 4 loaders),
so they're never used by the iPhone runtime. `scripts/strip-gemma-audio.py`
reads `model.safetensors` and writes a new file without the 754
`audio_tower.*` / `embed_audio.*` tensors — saves ~583 MB on disk with
zero functional impact.

The script keeps a `.audio.bak` copy as a safety net at
`scripts/backups/model.safetensors.audio.bak` — **outside** the bundle
resource path so Xcode's `type: folder` reference for `Resources/Models`
doesn't sweep it into the `.app`. (Critical: the very first run of the
strip script put the backup *inside* `Resources/Models/Gemma/`, which
bloated the `.app` bundle from ~3.1 GB to ~6.4 GB. The script now
defaults to `scripts/backups/` and migrates any legacy backup it finds.)

To restore:
```
mv scripts/backups/model.safetensors.audio.bak \
   HikeCompanion/Resources/Models/Gemma/model.safetensors
```
(or just re-run `bash scripts/fetch-gemma.sh` to pull a fresh copy.)

Vision-tower weights are kept because Phase 3b will turn them on.

## Phase status

- ✅ **Phase 1** — typed text → Gemma → Kokoro, multi-turn memory
- ✅ **Phase 2** — voice input via SFSpeechRecognizer (hold-to-speak)
- ✅ **Phase 3a** — camera capture via AVCaptureSession (real `UIImage`
  flowing through to GemmaService)
- ✅ **Phase 3b** — image Q&A via MLXVLM Gemma 4 multimodal. Working
  end-to-end on iPhone 17 Pro with the `increased-memory-limit`
  entitlement. Vision quality is OK but not great; foreground-object
  recognition is imperfect — see "Phase 3b VLM patches" below.

### Phase 3b VLM patches — what we changed in `mlx-swift-lm`

The Swift port of Gemma 4 in `mlx-swift-lm` has two issues for our use:

1. `Gemma4ProcessorConfiguration` reads top-level `size`/`image_mean`/
   `image_std`/`do_normalize`, but HF's official `processor_config.json`
   nests them under `image_processor`. Without hoisting, the Swift
   decoder defaults to 800×800 — blowing up memory.

2. `Gemma4Processor.preprocess` force-resizes to a single fixed square,
   while the Python reference does aspect-ratio-preserving dynamic
   sizing. At 224×224 (the HF default `size`) the patch grid is 14×14=196
   patches, and the Swift `Gemma4VisionPooler` degenerates: it returns
   196 raw + 84 zero outputs instead of 280 cleanly-pooled features.

**Workaround in `scripts/fetch-gemma.sh`**: post-fetch, the script
hoists the fields AND forces `size = 960×672` (portrait). At that shape,
the patch grid is 60×42 = 2520 (= `max_patches`) and the kernel=3 stride
pooler produces exactly 280 well-distributed pooled tokens, matching
training behavior. Memory is unchanged because the encoder always
processes 2520 tokens regardless.

**Instrumentation in `external/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift`**:
the `getInputEmbeddings` function logs `[Gemma4.mem]` at six checkpoints
during the forward pass (vision start, vision tower evaluated, vision
projection evaluated, vision cache cleared, image scatter evaluated,
language prefill returned). Useful for diagnosing forward-pass crashes;
also bounds peak memory by `eval()`-ing + `Memory.clearCache()` between
steps. **Strip when stable.**

**Open follow-up**: aspect-ratio-preserving resize. Currently a landscape
photo (wider than tall) gets vertically stretched into the 960×672
portrait frame, which hurts recognition. The clean fix is to port the
Python reference's `aspect_ratio_preserving_resize` to Swift — non-trivial
but matches training distribution.

## Known gotchas

- **Build for iOS Simulator** works for compile verification but **the app
  cannot run on Simulator** — MLX requires Metal compute that the simulator
  doesn't have.
- **iPhone needs Developer Mode enabled** (Settings → Privacy & Security →
  Developer Mode → on, then reboot) before any sideloaded build can launch.
- **Free Apple ID dev certs expire after 7 days** — re-run from Xcode each
  week if not on a paid Developer Program account.
- **Release build doesn't work** out of the box — Xcode 26's strict module
  scanner fails on transitive deps (`Atomics`, `DequeModule`, `Numerics`).
  Use Debug. To reduce Debug overhead: scheme → Run → Diagnostics → uncheck
  Main Thread Checker and Thread Performance Checker.
- **TTS on long input glitches** if not chunked — KokoroSwift's duration
  predictor goes unstable past ~60 chars. `splitForSynthesis` in
  ValidationRunner splits on `. ! ? , : ;` with `maxCharsPerChunk = 80`.
  **Don't raise above 80**; below ~60 chars the model behaves.
- **VLM Q&A requires the `increased-memory-limit` entitlement.** Without
  it, the iPhone foreground jetsam line (~3.5 GB) is below the VLM peak
  (~3.54 GB), and the app silently dies during prefill eval. See
  "iOS jetsam entitlement" above.
- **VLM image preprocessing forces 960×672 portrait shape.** Square
  photos and landscape photos get distorted. This is the simplest fix
  that matches the trained pooler — see "Phase 3b VLM patches".
- **`[Mem]` ticker spams the Xcode console.** Filter to `[Mem]` to see
  only memory profile lines. The 5 s ticker is started in
  `HikeCompanionApp.init()`.

## What survived debugging — pointers to commits

- `6bf3103` — embed KokoroSwift dynamic framework (dyld __abort_with_payload fix)
- `aacbabb` — preserve Models/Gemma/ tree in bundle (Gemma loader weight collision fix)
- `b7fea0e` — Bundle.url subdirectory: "Models" (Kokoro lookup fix)
- `e681ee2` — re-add gemma.unload() between generation and TTS (jetsam fix)
- `423d207` — cap conversation history at 20 messages (10 turns)
- `d8d001e` — two-phase serial Kokoro unload (timing fix; this is subtle)
- `fa3a69b` — Kokoro status reflects "Idle" when unloaded
- `ad11b9b` — hold-to-speak gesture, auto-fires Ask on release
- `849b543` — UI redesign + Phase 3a camera + Phase 3b VLM scaffolding
- `4b82876` — Phase 3b VLM actually works on device (entitlement,
  image-size patch, vendored mlx-swift-lm, instrumentation)

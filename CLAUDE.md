# CLAUDE.md — hikeCompanion

Context checkpoint for future Claude sessions. Read this first.

## What this is

iOS app for the **Kaggle Gemma 4 for Good** hackathon. The product idea is a
"hike companion" that explains what hikers see in front of them — geology,
plants, climate change in national parks. Everything runs **on-device**:

- **Gemma 4 E2B** (INT4, ~3.5 GB on disk, ~2.5 GB MLX active) for the LLM
- **Kokoro 82M** (FP32 safetensors, ~327 MB) for TTS
- **SFSpeechRecognizer** (Apple, on-device) for voice input

User flow today: **press-and-hold mic → speak question → release →
Gemma streams response → Kokoro speaks it aloud**. Multi-turn memory
preserved across 10 conversation turns.

## Repo

- GitHub: `git@github.com:YingCeci/hikeCompanion.git` (private — under
  YingCeci, not lijuncheng16, due to the SSH key on this Mac)
- Local path: ``
- Owner intent: Billy Li (`lijuncheng16`); collaborator: Ying Wang (`YingCeci`).
  Bundle ID is `com.lijuncheng16.HikeCompanion`.

## Tech stack — what's vendored and why

`external/kokoro-ios`, `external/MisakiSwift`, `external/MLXUtilsLibrary` are
all **vendored copies** with patched `Package.swift` files. **Do not delete
or replace with URL-based deps** without understanding the conflict:

- **mlalma's KokoroSwift 1.0.11** (the upstream we're based on) hard-pins
  `mlx-swift exact: "0.30.2"`. So do MisakiSwift 1.0.6 and MLXUtilsLibrary 0.0.6.
- **mlx-swift-lm 3.x** (the only version with Gemma 4 support) requires
  `mlx-swift 0.31+`. Direct conflict.
- Fix: vendor all three mlalma packages, relax their MLX pins to ranges,
  use sibling-relative `path:` references so they all resolve together.
- **`MLXUtilsLibrary 0.0.7+` removed `BenchmarkTimer`** but KokoroSwift 1.0.11
  still calls it. We re-added a no-op stub at
  `external/MLXUtilsLibrary/Sources/MLXUtilsLibrary/Utils/BenchmarkTimer.swift`.
  Don't delete this file.

URL-based SPM deps (in `project.yml`):
- `mlx-swift-lm` ≥ 3.31.3
- `swift-transformers` ≥ 1.3.0 — products `Tokenizers`, `Hub`
- `swift-huggingface` ≥ 0.8.1 — product `HuggingFace` (needed for the
  `#huggingFaceTokenizerLoader()` macro to compile)

Macros require **explicit trust on first Xcode open** ("Trust & Enable All").
For CLI builds, pass `-skipMacroValidation`.

## Critical lifecycle patterns — DO NOT REGRESS

### Gemma is lazy-loaded per Ask, unloaded after generation

`GemmaService.loadIfNeeded()` is called at the start of every Ask;
`gemma.unload()` runs after generation completes. This:

- Pays a 10–30 s reload per Ask (model file mmap + MLX kernel JIT).
- **Bounds memory**. Keeping Gemma resident across the Gemma → Kokoro
  hand-off OOM'd the app even on iPhone 17 Pro.
- **Conversation history persists in `GemmaService` itself**, not in the
  ModelContainer — survives unload/reload. Replayed into a fresh
  `ChatSession` per call. Cap: `maxHistoryMessages = 20` (10 turns).

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

| State | Process footprint | MLX active | MLX peak |
|---|---|---|---|
| Cold start | 41 MB | 14 MB | 14 MB |
| Idle between Asks | ~100 MB | 14 MB | (lifetime) |
| Gemma loaded | ~2.7 GB | 2.47 GB | 2.97 GB |
| Generation | ~2.8 GB | 2.47 GB | 2.97 GB |
| After Gemma unload | ~1.5 GB → 100 MB | **14 MB** | 2.97 GB |

iOS jetsam threshold ~5–6 GB on iPhone 17 Pro. Headroom ~2 GB.
Both models successfully unload between turns (MLX active = 14 MB).

## Project layout

```
hikeCompanion/
├── README.md, CLAUDE.md, project.yml, .gitignore
├── scripts/
│   ├── fetch-models.sh       # Kokoro safetensors + voices.npz from KokoroTestApp Git LFS
│   ├── fetch-gemma.sh        # Gemma 4 E2B INT4 from mlx-community/gemma-4-e2b-it-4bit
│   └── generate-project.sh   # xcodegen wrapper
├── external/
│   ├── kokoro-ios/           # vendored, MLX pin relaxed
│   ├── MisakiSwift/          # vendored, MLX pin relaxed, sibling path: dep
│   └── MLXUtilsLibrary/      # vendored, BenchmarkTimer stub re-added
└── HikeCompanion/
    ├── HikeCompanionApp.swift     # @main, MLX cache limit only
    ├── ContentView.swift          # SwiftUI: hold-to-speak, Ask, debug TTS, memory profiler
    ├── GemmaService.swift         # lazy-load + unload + history replay
    ├── ValidationRunner.swift     # Kokoro wrapper, two-phase unload, chunked TTS
    ├── SpeechRecognizer.swift     # SFSpeechRecognizer wrapper
    ├── MemoryStats.swift          # task_vm_info + MLX.Memory.snapshot
    ├── Info.plist (generated)
    ├── Assets.xcassets/
    └── Resources/Models/          # gitignored — fetch via scripts above
```

## Setup commands (cold clone)

```bash
git clone --recurse-submodules git@github.com:YingCeci/hikeCompanion.git
# (no actual submodules — `external/` is committed directly — but keeping
# the recurse flag harmless if we ever switch back)
cd hikeCompanion

bash scripts/fetch-models.sh    # Kokoro: ~630 MB
bash scripts/fetch-gemma.sh     # Gemma 4 E2B: ~3.5 GB. Add --backup for unsloth fallback.
bash scripts/generate-project.sh

open HikeCompanion.xcodeproj
# In Xcode: trust macros when prompted; set Development Team in Signing & Capabilities
# ⌘R to a real iPhone (≥ iPhone 15 Pro / iOS 18). Simulator does not have MLX.
```

## Phase status

- ✅ **Phase 1** — typed text → Gemma → Kokoro, multi-turn memory
- ✅ **Phase 2** — voice input via SFSpeechRecognizer (hold-to-speak,
  auto-fires Ask on release)
- ⬜ **Phase 3** — camera / image input via MLXVLM Gemma 4 multimodal
  (not started)

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

## What survived debugging — pointers to commits

- `6bf3103` — embed KokoroSwift dynamic framework (dyld __abort_with_payload fix)
- `aacbabb` — preserve Models/Gemma/ tree in bundle (Gemma loader weight collision fix)
- `b7fea0e` — Bundle.url subdirectory: "Models" (Kokoro lookup fix)
- `e681ee2` — re-add gemma.unload() between generation and TTS (jetsam fix)
- `423d207` — cap conversation history at 20 messages (10 turns)
- `d8d001e` — two-phase serial Kokoro unload (timing fix; this is subtle)
- `fa3a69b` — Kokoro status reflects "Idle" when unloaded
- `ad11b9b` — hold-to-speak gesture, auto-fires Ask on release

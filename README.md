# hikeCompanion — Kokoro ANE Validation Harness

A minimal iOS app that runs [mattmireles/kokoro-coreml](https://github.com/mattmireles/kokoro-coreml) on iPhone to answer one question:

**Does the Apple Neural Engine path deliver acceptable text-to-speech latency on real iPhone hardware?**

This is a throwaway harness, not a product. The goal is to produce two numbers — RTF on `.all` (ANE-preferred) and RTF on `.cpuAndGPU` (no ANE) — so we can decide whether to commit to mattmireles' Core ML pipeline for the full hike companion app, or fall back to [mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios) (MLX/GPU, proven on iPhone).

## Why this exists

Read the upstream README first if you haven't: https://github.com/mattmireles/kokoro-coreml

The upstream's published numbers are all from **Mac** Apple Silicon (M1, M2, M2 Ultra). There are no iPhone benchmarks. This harness fills that gap.

## Architecture

The upstream Swift package (`KokoroPipeline`) takes pre-tokenized inputs (`input_ids`, `attention_mask`, `ref_s` voice embedding) and produces audio. There is no Swift G2P/phonemizer in the upstream — the text-to-input_ids step lives in Python (`scripts/prepare_swift_bench_inputs.py`).

So the validation flow is:

```
[Mac, one-time]
  prepare-fixtures.sh
    └─ runs upstream prepare_swift_bench_inputs.py
       └─ writes JSON fixtures (3s.json, 7s.json, 15s.json, 30s.json, hnsf_weights.json)
          to HikeCompanion/Resources/Fixtures/
fetch-models.sh
  └─ downloads .mlpackage files from Hugging Face
     to HikeCompanion/Resources/Models/

[iPhone, every run]
  HikeCompanion app
    1. Load JSON fixture from bundle
    2. Init KokoroPipeline pointing at bundled Models/ dir
       with chosen MLComputeUnits (.all vs .cpuAndGPU)
    3. Call executeKokoroSynthesis with the fixture's input_ids
    4. Measure wall time, audio duration, compute RTF
    5. Save WAV to Documents/, share via UIActivityViewController
```

## Quick start

### Prerequisites
- macOS with Xcode 26+ (deployment target iOS 17+)
- Python 3.10+ with `uv` (https://docs.astral.sh/uv/)
- A real iPhone (the Simulator does not have an ANE)
- Apple Developer account / Team ID (for code signing)

### Setup

```bash
git clone --recurse-submodules https://github.com/lijuncheng16/hikeCompanion.git
cd hikeCompanion

# 1) Generate JSON fixtures (one-time, ~5 min, runs Python)
bash scripts/prepare-fixtures.sh

# 2) Download Core ML models from Hugging Face (~200 MB)
bash scripts/fetch-models.sh

# 3) Generate the Xcode project
bash scripts/generate-project.sh
```

Then open `HikeCompanion.xcodeproj` in Xcode, set your Development Team in target settings, and build to a real device.

### What success looks like

- App launches, shows compute-units picker and fixture picker.
- Tap "Run Once" → first run takes ~30s (model compilation), subsequent runs are fast.
- Display shows: wall time, audio duration, **RTF** (real-time factor, lower is better — < 1.0 means faster than real-time).
- "Play" button plays the synthesized audio. Should sound like clean speech, not glitched/garbled.
- "Share WAV" exports the file via AirDrop/Mail/Files.

### What failure looks like (and the fallback)

- **Build fails** with iOS-incompatible API in `KokoroPipeline` → upstream is Mac-leaning despite declaring iOS 16 platform. Fall back to mlalma/kokoro-ios.
- **App crashes on init** with "model not loaded" / OOM → INT8 model is too large for the device. Try a smaller bucket only.
- **RTF > 1.0 on `.all`** but **RTF < 1.0 on `.cpuAndGPU`** → ANE is producing wrong output and Core ML is silently falling back to CPU. Use `.cpuAndGPU` and accept the GPU contention with Gemma.
- **Audio sounds glitched on `.all`** but clean on `.cpuAndGPU` → known ANE-compiler issue from upstream README. Use `.cpuAndGPU`.
- **Both modes fail or are slow** → mattmireles is not viable on iPhone. Use mlalma.

## Compute units to test

| Mode | What it means |
|---|---|
| `.all` | Default. Core ML decides; prefers ANE for compatible ops. |
| `.cpuAndNeuralEngine` | ANE + CPU only, no GPU. Pure ANE test. |
| `.cpuAndGPU` | No ANE — compares against mlalma's MLX/GPU baseline. |
| `.cpuOnly` | Sanity floor. Should be slowest. |

Run each fixture (3s, 7s, 15s, 30s) on each mode. Record RTF. Listen to each WAV.

## Layout

```
hikeCompanion/
├── README.md                              # this file
├── project.yml                            # xcodegen spec → .xcodeproj
├── scripts/
│   ├── fetch-models.sh                    # downloads .mlpackage from HF
│   ├── prepare-fixtures.sh                # runs upstream Python prep
│   └── generate-project.sh                # xcodegen wrapper
├── external/
│   └── kokoro-coreml/                     # git submodule (upstream)
└── HikeCompanion/
    ├── HikeCompanionApp.swift             # @main
    ├── ContentView.swift                  # SwiftUI: pickers + run buttons
    ├── ValidationRunner.swift             # load fixture → synth → time → save
    ├── ConfigurableModelProvider.swift    # KokoroModelProvider impl with chosen MLComputeUnits
    ├── BenchTypes.swift                   # JSON Decodable structs (matches upstream)
    ├── WAVWriter.swift                    # mono 16-bit PCM WAV (vendored from upstream)
    ├── Info.plist
    ├── Assets.xcassets/
    └── Resources/
        ├── Fixtures/                      # JSON inputs (generated by prepare-fixtures.sh)
        └── Models/                        # .mlpackage files (downloaded by fetch-models.sh)
```

## Out of scope

- Text → audio (no phonemizer). Use mlalma/kokoro-ios for that, or wire `MisakiSwift` in later.
- Streaming output (synthesis is non-autoregressive but there's no chunked playback path here).
- Production UX. Buttons and a results label, that's it.

## License

App code: MIT. Upstream submodule: see `external/kokoro-coreml/LICENSE`.

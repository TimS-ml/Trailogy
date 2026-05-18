# iOS App Architecture (Trailogy)

## TL;DR

- Trailogy is an offline iOS trail guide that runs language, vision, speech, and retrieval models on the device with no runtime network dependency.
- SwiftUI views call long-lived services owned by the root view, and only the walking experience drives model inference.
- The app bundle includes the model files, trail content, retrieval corpus, and image assets needed for normal use.
- The public app name is Trailogy, while some internal target and source names remain unchanged to avoid provisioning churn.

## Product summary

Trailogy is an iOS app for the Kaggle Gemma 4 for Good hackathon. A
self-guided audio trail guide for three trails: walks the user through
pre-authored stop narration, accepts voice + photo questions, and
recalls grounded answers via on-device retrieval. **Everything runs
on-device with zero network at run time:**

- **Gemma 4 E2B** (INT4, ~2.8 GB) — LLM / VLM. Optionally swappable
  to the SFT'd finetune produced by the model-side pipeline.
- **Kokoro 82M** (FP32, ~310 MB active) — Text-to-Speech.
- **SFSpeechRecognizer** (Apple, on-device) — Speech-to-Text.
- **MiniLM-L6-v2** (FP32, ~87 MB) — sentence embeddings for RAG retrieval.
- **Trail content + RAG corpus + cover/stop photos** bundled in the
  `.app` (no first-launch download).

Total on-device footprint: **~3.4 GB** installed
(Gemma INT4 ~2.8 GB + Kokoro ~0.31 GB + MiniLM ~0.09 GB + RAG corpus
~0.02 GB + Swift binaries + assets). Doubles to ~6.2 GB if the user
keeps the stock Gemma backup in `Models/Gemma.stock/` alongside an
SFT'd finetune in `Models/Gemma/` (Phase 7 fetch-and-backup flow).

## High-level architecture

### Layer 1 — Views → Services (who calls what)

```mermaid
graph TD
    subgraph Views
        WV[WalkingView]
        DV[DetailView]
        PV[PickerView]
        JV[JournalView]
        DbV[DebugView]
    end
    subgraph Services
        GS[GemmaService]
        VR[ValidationRunner]
        SR[SpeechRecognizer]
        CC[CameraController]
        RS[RAGService]
        IS[ImageStore]
        AR[AppRouter]
    end

    WV --> GS
    WV --> VR
    WV --> SR
    WV --> CC
    WV --> RS
    WV --> AR
    DV --> AR
    DV --> IS
    PV --> AR
    PV --> IS
    JV --> AR
    DbV --> GS
    DbV --> VR
    DbV --> RS
    DbV --> AR
```

`ContentView` owns all services (`GemmaService`, `ValidationRunner`,
`SpeechRecognizer`, `RAGService`, `AppRouter`, `ImageStore`) as
`@StateObject`. `AppRouter` drives navigation. **`WalkingView` is the
only view that drives ML inference**; the other views read shared
state (downloadedTrailIDs, walkedAt, RAG subjects override) and
render UI.

### Layer 2 — Services → Frameworks (what powers what)

```mermaid
graph TD
    subgraph Services
        GS[GemmaService]
        VR[ValidationRunner]
        SR[SpeechRecognizer]
        CC[CameraController]
        RS[RAGService]
        IS[ImageStore]
    end
    subgraph Frameworks
        MLXLLM[MLXLLM]
        MLXVLM[MLXVLM]
        KS[KokoroSwift]
        SE[swift-embeddings]
        SF[SFSpeechRecognizer]
        AV[AVFoundation]
        MK[MapKit]
        FM[FileManager]
        URLS[URLSession]
    end
    subgraph Runtime
        MLX[MLX Swift → Metal]
    end

    GS --> MLXLLM
    GS --> MLXVLM
    VR --> KS
    SR --> SF
    CC --> AV
    RS --> SE
    IS --> FM
    IS --> URLS
    MLXLLM --> MLX
    MLXVLM --> MLX
    KS --> MLX
    SE --> MLX
```

**Four MLX consumers** (Gemma, Kokoro, MiniLM, vendored shaders) share
one Metal runtime, one allocator, one cache. Hand-off discipline
(unload-before-load, two-phase serial unload, scenePhase gates — see
[`03-memory-management.md`](03-memory-management.md) and
[`06-scenephase-metal-background.md`](06-scenephase-metal-background.md))
is the architectural fact that makes the entire stack fit.

## App state machine (AppRouter)

```mermaid
stateDiagram-v2
    [*] --> picker
    picker --> detail : selectTrail(trail)
    detail --> picker : back
    detail --> walking : begin (after demo-alert)
    walking --> journal : endTour
    journal --> picker : done
    walking --> walking : askQuestion / narrate / camera / advancePhase
```

States:
- **picker** — Trail selection home (3 cards). Each card carries
  downloaded + completed badges.
- **detail** — Trail info + state-aware Download CTA + Begin button
  (gated by demo-mode alert).
- **walking** — Active tour. Has its own internal `TourPhase` machine
  (below).
- **journal** — Post-tour Recap (Discoveries stream).

### Internal tour state machine (`WalkingView.TourPhase`)

```mermaid
stateDiagram-v2
    [*] --> atStop : starts at stop 1
    atStop --> between : narration finishes, before approaching
    between --> approaching : APPROACHING_MS countdown
    approaching --> atStop : ARRIVAL_MS countdown, advance stopIdx
    atStop --> complete : last stop's atStop window ends
    complete --> [*] : router.endTour() → .journal
```

- **atStop** — narration playing (Kokoro streaming) + look-for /
  payoff arc rendered + mic / camera / more visible.
- **between** — walking-dots animation, prev stop's lookFor visible.
- **approaching** — heavy haptic, lime walking-dots + glow.
- **complete** — terminal; completion card + "Open journal" CTA.

Backgrounding the app from `.atStop` / `.approaching` halts Kokoro
synth via `scenePhase` `.onChange` — prevents Metal command-buffer
submission while the app is backgrounded (see
[`06-scenephase-metal-background.md`](06-scenephase-metal-background.md)).

## Core data flow — Ask pipeline (with RAG injection)

```mermaid
sequenceDiagram
    participant User
    participant WV as WalkingView
    participant SR as SpeechRecognizer
    participant RS as RAGService
    participant GS as GemmaService
    participant VR as ValidationRunner
    participant Speaker as AVAudioEngine

    User->>WV: Hold mic button
    WV->>SR: startListening()
    SR-->>WV: partial transcripts (live)
    User->>WV: Release mic button
    WV->>SR: stopListening()
    SR-->>WV: final transcript

    WV->>RS: retrieve(query, topK=3)
    Note over RS: MiniLM stays resident<br/>(~87 MB, preloaded at .active)
    RS-->>WV: chunks grouped by subject
    WV->>WV: formatChunksForPrompt → gemma.ragContext

    WV->>GS: loadIfNeeded(.text)
    Note over GS: ~10-30 s on cold launch<br/>~0 s if Gemma was resident
    GS-->>WV: ready

    WV->>GS: stream(prompt + ragContext + [camera=off] gate)
    loop Token streaming
        GS-->>WV: token chunk
        WV-->>VR: queue chunk for TTS
    end

    Note over GS: gemma.unload() before TTS<br/>Memory.clearCache()

    VR->>VR: ensureModelLoaded() (Kokoro)
    loop Per chunk (max 80 chars)
        VR->>VR: synthesize PCM
        VR->>Speaker: schedule buffer
    end
    Speaker-->>User: Audio playback

    VR->>VR: Phase 2 (serial): tts = nil → clearCache()
```

## Core data flow — VLM (image) Ask

```mermaid
sequenceDiagram
    participant User
    participant WV as WalkingView
    participant CC as CameraController
    participant GS as GemmaService
    participant VR as ValidationRunner

    User->>WV: Tap camera button
    Note over WV: First tap shows camera-tip alert<br/>(plants-only honesty)
    WV->>CC: capturePhoto()
    CC->>CC: downscale to 1280 px → flash if needed
    CC-->>WV: UIImage

    User->>WV: Ask about photo
    WV->>GS: loadIfNeeded(.vlm)
    Note over GS: Loads vision tower + language model<br/>~3.54 GB peak prefill

    WV->>GS: stream([camera=on] prompt, image, history=empty)
    Note over GS: 280 image tokens + system + user<br/>[camera=on] SFT data-prefix gate
    GS-->>WV: streamed response

    WV->>GS: unload()
    WV->>WV: capturedImage = nil (free ~3 MB)
    WV->>VR: synthesize response
```

VLM path **bypasses RAG** (image-Q is closed-form on the photo
content) and **bypasses chat history** (cold context to keep KV cache
small). `[camera=on]` is emitted by `GemmaService` to match the
training-time input marker (Track B v4) — see
[`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md)
for why preserving the SFT input marker at inference matters.

## Core data flow — tour-guide narration (streaming TTS)

```mermaid
sequenceDiagram
    participant Timer as Phase timer
    participant WV as WalkingView
    participant RS as RAGService
    participant GS as GemmaService
    participant VR as ValidationRunner

    Timer->>WV: phase=.atStop (arrival)
    WV->>WV: spokenNarrationWithPayoff<br/>(prepend prev stop's payoff)
    WV->>VR: ensureModelLoaded()
    loop For each narration sentence
        WV->>VR: synthesize → schedule
        VR-->>Timer: starts speaking
    end

    Note over WV: lookFor for THIS stop posted to<br/>between/approaching panel

    Timer->>WV: phase=.between (after narration)
    WV->>WV: show walking-dots + lookFor panel
    Timer->>WV: phase=.approaching
    WV->>WV: lime walking-dots + heavy haptic prep
    Timer->>WV: phase=.atStop (next stop) — heavy haptic
    WV->>WV: stopIdx += 1, loop
```

Narration is RAG-augmented per-stop (the active subjects set is
populated from `trail.defaultRAGSubjects` at tour start, then
narration prompts pull top-k chunks). User Asks (mic or camera)
interrupt narration; on resume, the remaining sentences continue from
the interrupt point.

## Core data flow — offline image cache

```mermaid
sequenceDiagram
    participant User
    participant DV as DetailView
    participant AR as AppRouter
    participant IS as ImageStore
    participant Net as URLSession

    User->>DV: Tap "Download · 68 MB"
    DV->>DV: CTA → .downloading

    loop For each image (cover + 5 stops)
        DV->>IS: downloadAll(trail, progress:)
        IS->>Net: fetch URL
        Net-->>IS: image bytes
        IS->>IS: write atomically to<br/>AppSupport/TrailImages/{id}/{cover|stop-N}.bin
        IS-->>DV: progress(i/6)
    end

    DV->>AR: markDownloaded(trail.id)
    DV->>DV: CTA → .ready ("Begin")

    Note over IS,AR: Next launch:<br/>AppRouter.init seeds downloadedTrailIDs<br/>from ImageStore.hasAllLocal — disk IS the persistence layer
```

`CachedTrailImage` is the drop-in for `AsyncImage`. Tries
`ImageStore.loadLocal` first; falls back to network `AsyncImage`.
`.task(id:)` re-checks disk so images downloaded after first render
take over without manual invalidation.

## Service responsibilities

| Service | Model / Backing | Active mem | Role |
|---|---|---|---|
| `GemmaService` | Gemma 4 E2B INT4 (stock or SFT) | ~2.5 GB text, ~3.5 GB peak VLM prefill | Text gen, multimodal vision, `[camera=on/off]` gate, offline-aware system prompt |
| `ValidationRunner` | Kokoro 82M FP32 | ~310 MB active | TTS synthesis with streaming chunks |
| `SpeechRecognizer` | Apple built-in | 0 MB (system) | On-device ASR |
| `RAGService` | MiniLM-L6-v2 + bundled corpus | ~87 MB resident | Multi-subject top-k retrieval, query-once-scan-all |
| `CameraController` | — | — | Photo capture + 1280-px downscale + auto-flash |
| `ImageStore` | Disk cache + URLSession | ~9 MB/trail on disk | Per-trail cover + stop image cache, atomic write |
| `AppRouter` | In-memory state | trivial | Nav + downloadedTrailIDs + walkedAt + ragSubjectsOverride |

## RAG architecture (preview)

Full details in [`05-rag-runtime.md`](05-rag-runtime.md). Short
version:

- MiniLM (~87 MB FP16, bundled — no first-launch HF download)
  preloaded at `.active`, resident across the Gemma load/unload cycle.
- 4 subject corpora (geology / plants / physics / english), each ~25
  hand-authored chunks, pre-embedded at build time, bundled in `.app`.
- Per-trail `defaultRAGSubjects` selects which corpora are active;
  DebugView can override at runtime.
- Per-Ask: embed query (~5 ms) → cosine vs every active subject's
  chunks (~1 ms) → top-k → `formatChunksForPrompt` groups by subject
  → injected into Gemma's user prompt one-shot.

### RAG vs VLM mode selection

| Ask type | Path | RAG injected? | History? |
|---|---|---|---|
| Text Q (mic) | `.text` | Yes (active subjects) | 20 messages (10 turns: 10 user + 10 assistant) |
| Image Q (camera) | `.vlm` | **No** (cold context) | **None** (cold context) |
| Tour narration (auto) | `.text` | Yes (active subjects) | None per-stop |

## Memory discipline (preview)

Full numbers in [`03-memory-management.md`](03-memory-management.md).
Short version:

```mermaid
sequenceDiagram
    participant App
    participant Mem as MLX Memory
    participant G as Gemma
    participant K as Kokoro
    participant M as MiniLM

    Note over App: App launch
    App->>Mem: configure cache limit
    App->>M: preload (lazy, on .active)
    Note over M: ~87 MB resident, stays put

    Note over App: First Ask
    App->>G: loadIfNeeded(.text)
    Note over G: ~2.5 GB allocated
    G-->>App: stream done
    App->>G: unload()
    App->>Mem: clearCache()
    Note over G,Mem: Memory back to baseline + MiniLM

    Note over App: TTS phase
    App->>K: load + synthesize
    Note over K: ~310 MB allocated
    K-->>App: speaks
    App->>K: tts = nil  (Phase 1)
    App->>Mem: clearCache()  (Phase 2, serial)
    Note over K,Mem: Two-phase serial unload — ARC + cache race fix
```

**The three hand-off discipline rules** (each born from a real Phase
1 bug):

1. **Unload Gemma BEFORE TTS** — both resident > 5 GB → jetsam.
2. **Two-phase serial Kokoro unload** — `tts = nil` and
   `clearCache()` must be in separate `workQueue` phases; local
   closure bindings hold refs until the closure exits.
3. **scenePhase gate on every MLX allocator activity** — preload,
   synth, streaming all guarded against `.background` (see
   [`06-scenephase-metal-background.md`](06-scenephase-metal-background.md)).

## Project directory structure

```
Trailogy/                                # repo root (display name)
├── HikeCompanion/                       # source directory (legacy name)
│   ├── HikeCompanionApp.swift           @main, MLX config
│   ├── ContentView.swift                Root view, owns services
│   ├── AppRouter.swift                  Nav state machine + downloadedTrailIDs +
│   │                                    walkedAt + ragSubjectsOverride
│   ├── GemmaService.swift               LLM/VLM lifecycle, [camera=on/off] gate,
│   │                                    offline-aware system prompt
│   ├── ValidationRunner.swift           TTS lifecycle + chunking
│   ├── SpeechRecognizer.swift           ASR wrapper
│   ├── CameraController.swift           Photo capture + downscale + auto-flash
│   ├── RAGService.swift                 Multi-subject retrieval
│   ├── ImageStore.swift                 On-disk image cache
│   ├── MemoryStats.swift                Memory monitoring
│   ├── TrailData.swift                  Trail catalog + stops + learnings +
│   │                                    defaultRAGSubjects + lookFor/payoff +
│   │                                    CLLocationCoordinate2D coords
│   ├── Theme.swift                      Design tokens
│   ├── HikeCompanion.entitlements       Mic + Camera + Speech entitlements +
│   │                                    increased-memory-limit
│   ├── Info.plist                       CFBundleDisplayName=Trailogy + permissions
│   ├── Views/
│   │   ├── WalkingView.swift            Active tour
│   │   ├── PickerView.swift             Trail selection
│   │   ├── DetailView.swift             Trail info + Download CTA + MapKit preview
│   │   ├── JournalView.swift            Recap (Discoveries + trailmark + 9-cat icons)
│   │   ├── CameraView.swift             Camera modal
│   │   ├── CameraPreviewView.swift      AVCapturePreviewLayer
│   │   ├── TourMapView.swift            In-tour map overlay (MapKit)
│   │   ├── TrailMapShape.swift          Native MapKit TrailMapView
│   │   ├── CachedTrailImage.swift       AsyncImage + disk cache
│   │   └── DebugView.swift              Developer console + RAG subject picker
│   ├── Assets.xcassets/                 AppIcon (Trailogy logo, full-bleed)
│   └── Resources/                       (gitignored where noted)
│       ├── Models/                      (gitignored, fetched by scripts)
│       │   ├── kokoro-v1_0.safetensors
│       │   ├── voices.npz
│       │   ├── Gemma/                   stock or finetune (swappable)
│       │   │   ├── config.json
│       │   │   ├── model.safetensors
│       │   │   └── processor_config.json (patched to 960×672)
│       │   └── MiniLM/                  bundled, ~87 MB
│       │       ├── model.safetensors
│       │       ├── tokenizer.json
│       │       └── config.json
│       └── RAG/                         bundled corpus
│           ├── manifest.json
│           ├── geology.jsonl  + .embeddings.f16
│           ├── plants.jsonl   + .embeddings.f16
│           ├── physics.jsonl  + .embeddings.f16
│           └── english.jsonl  + .embeddings.f16
├── external/
│   ├── kokoro-ios/                      Vendored KokoroSwift
│   ├── MisakiSwift/                     Vendored G2P
│   ├── MLXUtilsLibrary/                 Vendored utils (+ BenchmarkTimer stub)
│   └── mlx-swift-lm/                    Local copy for VLM debugging
├── rag-poc/                             Embed pipeline (Python)
├── scripts/
│   ├── fetch-models.sh                  Download Kokoro weights
│   ├── fetch-gemma.sh                   Download stock Gemma + patch config
│   ├── fetch-gemma-finetune.sh          Pull SFT finetune from gated HF
│   ├── strip-gemma-audio.py             Remove audio tower (~583 MB saved)
│   ├── generate-project.sh              xcodegen wrapper
│   ├── embed-rag-corpus.py              MiniLM batch embed → f16
│   ├── query-rag-corpus.py              Retrieval debug REPL
│   ├── switch-gemma.sh                  Variant-management
│   └── rag-requirements.txt
├── project.yml                          XcodeGen project definition
└── README.md
```

## Technology stack

| Layer | Technology |
|---|---|
| UI | SwiftUI (iOS 18+), MapKit |
| LLM/VLM | `mlx-swift-lm 3.x` (MLXLLM / MLXVLM) |
| TTS | `KokoroSwift 1.0.11` (MLX-based) |
| STT | Apple `SFSpeechRecognizer` (on-device) |
| Embeddings | `swift-embeddings` (MiniLM-L6-v2, MLX backend) |
| ML Runtime | MLX Swift 0.31+ → Metal |
| Camera | AVFoundation `AVCaptureSession` |
| Audio | `AVAudioEngine` + `AVAudioPlayerNode` |
| Image cache | `FileManager` (Application Support) + `URLSession` |
| Build | XcodeGen + SwiftPM |
| Target | iPhone 15 Pro+ (arm64, iOS 18+) |
| App size | ~3.4 GB installed (Gemma 2.8 + Kokoro 0.31 + MiniLM 0.09 + corpus 0.02 + Swift binaries + assets); doubles if `Gemma.stock` backup is kept alongside a finetune |

## Cross-references

- Model-side architecture (producer of `Models/Gemma/`):
  [`01-architecture-model-pipeline.md`](01-architecture-model-pipeline.md)
- Memory management deep-dive:
  [`03-memory-management.md`](03-memory-management.md)
- Xcode build + SPM dependency vendoring:
  [`04-xcode-build-and-deps.md`](04-xcode-build-and-deps.md)
- RAG runtime details:
  [`05-rag-runtime.md`](05-rag-runtime.md)
- scenePhase × Metal background pattern:
  [`06-scenephase-metal-background.md`](06-scenephase-metal-background.md)
- Optimization catalog:
  [`07-optimizations-and-future.md`](07-optimizations-and-future.md)
- iOS dev timeline:
  [`09-dev-timeline-ios.md`](09-dev-timeline-ios.md)

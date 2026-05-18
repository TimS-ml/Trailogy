# Optimizations — Catalog of What Shipped & What's Open

## TL;DR

- This doc catalogs shipped optimizations before adding new knobs or re-tuning existing ones.
- Memory improvements focus on lazy loading, explicit unloads, cache clearing, and avoiding simultaneous large-model residency.
- Size and speed improvements include stripping unused model weights, bounding generation length, chunking speech, and downscaling photos.
- The remaining future work is separated from shipped behavior so readers can distinguish proven fixes from open ideas.

## Part 1 — what shipped (categorized)

### 1. Memory lifecycle management

| # | Optimization | Effect |
|---|---|---|
| 1.1 | Gemma lazy-load per Ask | Never loaded at launch; 14 MB cold idle |
| 1.2 | Gemma unload after generation | `modelContainer = nil` + `clearCache()` |
| 1.3 | Kokoro lazy-load per synthesis | Not loaded until speech needed |
| 1.4 | Kokoro two-phase serial unload | Phase 2 on serial queue after Phase 1 exits (see [`03-memory-management.md`](03-memory-management.md)) |
| 1.5 | `Memory.clearCache()` at Gemma → Kokoro transition | Drops transient buffers before next model |
| 1.6 | `Memory.clearCache()` at Kokoro exit | Drops TTS buffers after playback |
| 1.7 | `Memory.clearCache()` inside VLM vision pipeline | Prevents lazy graph accumulation |
| 1.8 | No `Memory.memoryLimit` set | Avoids allocation pressure spikes at hand-off |
| 1.9 | `Memory.cacheLimit = 100 MB` | Bounds retained cache between phases |
| 1.10 | Photo context nil'd after image answer | Frees ~3 MB promptly |
| 1.11 | Photo nil'd on error path too | No leak on failure |
| 1.12 | Tour end resets all state | History, photo, narration all freed |

### 2. Model size & loading

| # | Optimization | Effect |
|---|---|---|
| 2.1 | Audio tower strip (~583 MB removed) | 3.4 GB → 2.8 GB on disk |
| 2.2 | Dual-mode loader (text vs VLM) | Text path skips vision tower (~700 MB less) |
| 2.3 | MLXLLM sanitize filters vision+audio keys | Text mode never allocates vision tensors |
| 2.4 | MLXVLM sanitize filters audio keys only | VLM keeps vision, skips audio |
| 2.5 | Kokoro voices kept resident (14 MB) | Too small to bother unloading |
| 2.6 | Backup stored outside bundle path | Prevents doubling `.app` size |
| 2.7 | `increased-memory-limit` entitlement | Jetsam threshold ~4 GB → ~6 GB |

### 3. Generation parameters

| # | Optimization | Effect |
|---|---|---|
| 3.1 | `maxTokens: 120` (was 160) | Bounds output length and KV growth |
| 3.2 | `maxKVSize: 1024` (was 1280) | Caps KV cache allocation; ~250-token savings |
| 3.3 | `prefillStepSize: 128` | Limits per-step transient memory |
| 3.4 | History cap: 20 messages (10 turns) for text; 0 for VLM | Limits replayed prefill cost |
| 3.5 | History cap: 0 messages (VLM path) | Image tokens already consume most budget — see [`03-memory-management.md`](03-memory-management.md) |
| 3.6 | Stop-framing excluded from saved history | Prevents stale context wasting KV tokens |
| 3.7 | Regional context compressed (~110 → ~50 words) | Fewer tokens = less activation memory (~3 MB/tok) |
| 3.8 | "Climate change" scope dropped from persona | Narrower persona = shorter system prompt |
| 3.9 | Image instructions folded into baseInstructions | Pay ~50 tokens once, not conditionally per VLM turn |

### 4. TTS / audio

| # | Optimization | Effect |
|---|---|---|
| 4.1 | Streaming playback (per-chunk schedule) | 5-10 s → 1-2 s perceived latency |
| 4.2 | 80-char chunk cap | Prevents duration predictor instability |
| 4.3 | 3-stage chunking (sentence → clause → hard) | Natural prosody boundaries |
| 4.4 | Text normalization (dashes, quotes, ellipses) | Prevents G2P artifacts |
| 4.5 | Stop flag checked mid-synthesis loop | Early bail on interrupt |
| 4.6 | Narration interrupt/resume with remainder | No re-synthesis of spoken content |
| 4.7 | Audio session switching (record ↔ playback) | Prevents hardware conflicts |

### 5. Image & vision

| # | Optimization | Effect |
|---|---|---|
| 5.1 | Photo downscale to 1280 px | 36 MB → 3 MB per capture |
| 5.2 | Vision processor metadata patch (960×672) | Clean 280-token pool — see [`13-mlx-vision-input-parity.md`](13-mlx-vision-input-parity.md) |
| 5.3 | Staged `eval()` in VLM pipeline | Prevents lazy graph memory spike |
| 5.4 | Camera session start/stop on appear/disappear | Hardware held only when visible |
| 5.5 | Camera auto flash | Uses hardware flash when available |

### 6. Concurrency & timing

| # | Optimization | Effect |
|---|---|---|
| 6.1 | Serial work queue for TTS lifecycle | Guarantees Phase 1 → Phase 2 ordering |
| 6.2 | `[weak self]` in all async closures | Prevents retain cycles |
| 6.3 | Synchronous `isRunning = true` on main | Prevents race with `waitForTTS` |
| 6.4 | Phase timer suspension during interactions | Prevents narration/Ask race |
| 6.5 | Camera on dedicated serial queue | Off main thread |

### 7. Build & bundle

| # | Optimization | Effect |
|---|---|---|
| 7.1 | Blue-folder model separation | Prevents weight file collision (see [`04-xcode-build-and-deps.md`](04-xcode-build-and-deps.md)) |
| 7.2 | Vendored deps with relaxed MLX pins | Resolves 0.30 vs 0.31 conflict |
| 7.3 | `BenchmarkTimer` no-op stub | Satisfies KokoroSwift with zero cost |
| 7.4 | KokoroSwift embedded as dynamic framework | Fixes dyld load crash |
| 7.5 | MLXUtilsLibrary NOT embedded (transitive) | Prevents duplicate task error |
| 7.6 | On-device ASR (zero model cost) | No Whisper (~500 MB) needed |

### 8. Prompt engineering (memory-aware)

| # | Optimization | Effect |
|---|---|---|
| 8.1 | VLM escape-hatch for off-context photos | Prevents over-anchoring (coffee mug ≠ sandstone) |
| 8.2 | Prompt length treated as memory variable | ~3 MB activation savings per token trimmed |
| 8.3 | Regional context word count budgeted | Dense species lists replace prose framing |

### 9. RAG runtime

| # | Optimization | Effect |
|---|---|---|
| 9.1 | Tiny embedder (MiniLM all-MiniLM-L6-v2 FP16) | ~87 MB resident vs ~135 MB for nomic-embed |
| 9.2 | Embedder bundled in-app | No first-launch HF download; offline-from-install |
| 9.3 | Background preload via `.task(id: scenePhase)` | MiniLM ready before user reaches WalkingView |
| 9.4 | Subject corpora pre-embedded at build time | Zero runtime embedding cost for the corpus side |
| 9.5 | L2 normalization at ingest → dot product at retrieval | Cosine sim collapses to one multiply per dim |
| 9.6 | Float16 embeddings on disk, float32 in RAM | Halves bundle weight per corpus |
| 9.7 | Flat (brute-force) search over ~25 chunks | < 1 ms; ANN unnecessary until corpus > ~10K chunks |
| 9.8 | `ragContext` is one-shot, cleared after stream | Next turn re-retrieves rather than carrying stale context |
| 9.9 | k=1 chunk per Ask (vs k=2/3) | Stays safely inside 1024 maxKVSize budget |
| 9.10 | RAG skipped for VLM Asks | Image already costs ~280 soft tokens; stacking RAG would blow KV |
| 9.11 | `[RAG]` / `[Gemma]` tagged diagnostics | Filter-friendly Xcode console |

### 10. iOS background-execution safety

| # | Optimization | Effect |
|---|---|---|
| 10.1 | `.task(id: scenePhase)` gates MiniLM preload on `.active` | Prevents Metal-bg crash on prewarmed launch |
| 10.2 | `.onChange(of: scenePhase)` → `tts.stop()` on leave-`.active` | Kokoro halts before next chunk submit hits backgrounded app |
| 10.3 | `didPreloadRAG` one-shot latch | Preload fires exactly once over app lifetime |

See [`06-scenephase-metal-background.md`](06-scenephase-metal-background.md)
for the full pattern + residual risk.

### 11. Multi-subject RAG / offline / SFT gate / brand

| # | Optimization | Effect |
|---|---|---|
| 11.1 | Multi-subject RAG with eviction | `Set<Subject>` + per-trail defaults, diffs incoming vs loaded; bounded memory as user toggles |
| 11.2 | Per-trail `defaultRAGSubjects` | Trail catalog → curated subject sets; no service-layer coupling |
| 11.3 | DebugView RAG override picker | 4-toggle effective-set editor + reset-to-default |
| 11.4 | `[camera=on/off]` SFT data-prefix gate | Matches Track B v4 training input distribution (~3 tokens); prevents distribution drift on photo asks |
| 11.5 | Offline image cache (`ImageStore` + `CachedTrailImage`) | Cover + 5 stop images per trail to disk (~9 MB); `AppRouter.init` reseeds `downloadedTrailIDs` from disk |
| 11.6 | Tour `.complete` terminal state | Drops silent stop-1 wraparound; completion card derived from trail metadata |
| 11.7 | Look-for / payoff engagement arc | Per-stop arc; arrival narration prepends previous stop's payoff |
| 11.8 | Native MapKit per trail | Drops hand-coded SVG; ~10-25 MB MapKit cost off VLM hot path |
| 11.9 | Gemma offline-aware system prompt | Refuses on weather/news/time/GPS — prevents fabricating real-time data |
| 11.10 | `PRODUCT_NAME` + `LD_RUNPATH_SEARCH_PATHS` in `project.yml` | xcodegen 2.x doesn't auto-inject Xcode defaults |
| 11.11 | `-Wno-unused-const-variable` | Silences ~24 warning blocks/build from MLX Metal kernel headers |
| 11.12 | Brand rename via `CFBundleDisplayName` only | hikeCompanion → Trailogy display name; Xcode target / bundle id unchanged |

### 12. Model-side (Track C deploy artifact)

| # | Optimization | Effect |
|---|---|---|
| 12.1 | MLX 4-bit g128 quantization for shipped Gemma | 9.6 GB bf16 → ~3.3 GB on disk; ~0.7 GB jetsam headroom for VLM prefill |
| 12.2 | EoRA post-quant adapter (training-free) | r=64 closes +4.3 pp on M2 g64-affine (83.7 % → 88.0 %, within bf16 noise) without retraining; ~5 MB adapter; ships through QLoRALinear path |
| 12.3 | bf16 retained on `vision_tower` / `embed_vision` | Prevents bnb-NF4 70 % → 0.1 % vision-encoder collapse (mandatory; see [`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md) §5) |

**Total: ~80 discrete optimizations** across 12 categories.

## Part 2 — open / future directions

### A. Model-level

| Idea | Expected gain | Complexity | Risk | Status |
|---|---|---|---|---|
| **Gemma 2-bit quantization** | ~50 % size reduction | Medium (needs mlx-swift-lm support) | Quality degradation on factual/science content | **Explored** — `mixed_3_4` → 3.20 GB but ~7 % match. Direct 2-bit not viable on Gemma 4 today. |
| **Vision tower quantization** (separate from language) | ~200 MB reduction on VLM path | Low | Image understanding quality | **Not viable** — bnb-NF4 on `vision_tower` drops PlantNet 70 % → 0.1 %. `skip_modules` MANDATORY. vision_tower stays bf16. |
| **Speculative decoding** with smaller draft model | 2-3× generation speedup | High (needs draft model + library support) | Memory for 2nd model | **Open** — would need a Gemma-distilled draft. |
| **KV cache quantization** (INT8 KV) | ~50 % KV memory reduction → larger context | Medium | Slight quality loss on long context | **Open** — `mlx-swift-lm` doesn't expose this yet. Prompt compression bought ~700-900 MB instead. |
| **Prompt caching / prefix sharing** | Eliminate system prompt re-prefill (~330 tokens/turn) | Medium | Cache invalidation logic | **Open** — would benefit narration path. |
| **B.1 → MLX bridge** | Land 2.77 GB CUDA hybrid as MLX deploy artifact | Medium (`bridges/hf_gptq_to_mlx.py` is scaffolding only) | Per-key qconfig reconstruction edge cases | **Queued** — spec'd, not built. |

### B. Runtime / memory

| Idea | Expected gain | Complexity | Risk |
|---|---|---|---|
| **Keep Gemma resident between turns** (if device has > 12 GB) | Eliminate 10-30 s reload | Low (conditional on RAM detection) | OOM on devices with less RAM |
| **Adaptive history length** based on available memory | More context when safe, less when tight | Medium | Unpredictable behavior |
| **Model warm-up during UI navigation** | Start loading Gemma while user navigates to walking view | Low | Wasted load if user backs out |
| **`mmap` without full read** (lazy page faults) | Faster perceived load start | Low (MLX already does this partially) | Jetsam during random page-in |
| **Kokoro model caching across turns** | Eliminate 1-2 s TTS reload | Low (remove Phase 2 nil, add timeout) | +310 MB idle footprint |

### C. TTS / audio

| Idea | Expected gain | Complexity | Risk |
|---|---|---|---|
| **Opus/AAC compression of synthesized PCM** | Reduce audio buffer memory | Medium | Latency from encode/decode |
| **Voice cloning** | Better UX | High (training pipeline) | Quality, model size |
| **Reduce Kokoro to FP16** | 327 MB → ~164 MB | Low (convert script) | Slight quality change |
| **Parallel chunk synthesis** | ~40 % faster total TTS | Medium (concurrent MLX ops) | Memory spike, race conditions |
| **Pre-synthesize common phrases** (trail names, greetings) | Instant start for known content | Low | Storage, staleness |

### D. UX / perceived performance

| Idea | Expected gain | Complexity | Status |
|---|---|---|---|
| **Predictive loading** based on trail stop proximity | Model ready before user asks | Medium (GPS trigger) | **Open** — production app would be Core Location gated |
| **Batch narration pre-generation** | No wait between stops | Medium | **Open** — narration is already pre-authored; only Q&A needs Gemma. May not be needed. |
| **Gemma stream cancellation** on `scenePhase` leave-`.active` | Eliminates the one residual Metal-background crash window | High (needs upstream `mlx-swift-lm` cancellation hook) | **Open** — see [`06-scenephase-metal-background.md`](06-scenephase-metal-background.md) §Residual risk |

## Cross-references

- iOS architecture (services these optimizations live in):
  [`02-architecture-ios-app.md`](02-architecture-ios-app.md)
- Memory deep-dive: [`03-memory-management.md`](03-memory-management.md)
- RAG runtime: [`05-rag-runtime.md`](05-rag-runtime.md)
- Model-side optimization context: [`01-architecture-model-pipeline.md`](01-architecture-model-pipeline.md)
- Quantization detail: [`../quantization/README.md`](../quantization/README.md)

# Audio & TTS

> TL;DR — This is the detail doc behind the writeup's audio-first
> architecture paragraph. Trailogy keeps hikers' eyes up by turning
> retrieved Gemma answers into spoken narration. We use Kokoro 82M on
> MLX because Gemma 4's small-model audio decoder was unavailable,
> Apple's system TTS was not natural enough for the product feel, and
> the first CoreML Kokoro path was too slow. Sentence-chunk streaming
> makes playback start in 1-2 s instead of waiting for a full response.

## Why audio-first

The interaction model is audio-first so hikers can stay focused on the
trail and engage with the experience with minimal screen attention.
The whole pipeline — hold-to-speak ASR, Gemma generation, RAG
retrieval, narration — funnels into a single spoken response so the
phone can stay in a pocket through most of the walk.

## What we did not use, and why

### Gemma 4's native audio decoder

Gemma 4 ships with a speech head, but it was not exposed for the E2B
(2B-parameter) variant we targeted — only larger variants had the
audio decoder available at the time. Bringing the audio path back
ourselves would have meant either (a) fine-tuning a full speech-to-
speech model from scratch, or (b) waiting on upstream. Both were out
of scope for the hackathon window, so we kept Gemma as a pure text
model and wired a dedicated TTS engine in front of it.

### Apple `AVSpeechSynthesizer` (system TTS)

Apple's built-in synthesizer is free and zero-bytes, but the default
voices sound flat and robotic against the rest of the UX — the
companion is supposed to feel like a knowledgeable friend on the
trail, not a turn-by-turn navigation prompt. Apple's premium neural
voices exist but require an opt-in download per language, vary by iOS
version, and still don't match modern open-source neural TTS on
prosody. For a demo where voice quality is core to the product
feeling, system TTS was the wrong floor.

### CoreML-packaged Kokoro

Our first cut bundled Kokoro converted to CoreML
([`4eb7fd3`](../../docs/general/09-dev-timeline-ios.md#phase-0--kokoro-tts-validation-may-2)).
Two things killed it:

1. **Too slow.** End-to-end synthesis of a single sentence took
   multiple seconds on-device — fine for batch generation, fatal for
   interactive narration where the hiker is waiting on every reply.
2. **No streaming path.** The CoreML graph was wrapped as a one-shot
   "text in, full audio out" call. There was no clean way to emit
   audio for partial input as Gemma generated, which is the only way
   to hide multi-second model latency behind playback (see [Streaming
   implementation](#streaming-implementation) below).

A secondary cost: running a second neural runtime (CoreML) alongside
the MLX runtime Gemma already needed meant duplicated Metal state, two
weight loaders to manage, and more places for memory pressure to spike
on iPhone.

## What we chose: Kokoro 82M on MLX

We pivoted to the
[mlalma/KokoroTestApp](https://github.com/mlalma/KokoroTestApp) MLX
port, vendored as a dynamic `KokoroSwift` framework
([`6bf3103`](../../docs/general/09-dev-timeline-ios.md#phase-0--kokoro-tts-validation-may-2)).

| | CoreML Kokoro | **MLX Kokoro (chosen)** | AVSpeechSynthesizer |
|---|---|---|---|
| Voice quality | Good | **Good (Kokoro 82M)** | Mediocre (basic) / Good (premium, opt-in) |
| Per-sentence latency | Multi-second | **~200–400 ms** | Low |
| Streaming support | None | **Per-chunk schedulable** | N/A (engine-internal) |
| Runtime collision with Gemma | Two runtimes (CoreML + MLX) | **Single MLX/Metal runtime** | Separate audio engine |
| Bundle cost | ~330 MB | **~340 MB** (model 327 MB + voices 14 MB) | 0 MB (or +30–100 MB per premium voice) |
| Memory hand-off control | Opaque | **Explicit MLX cache control** | N/A |

Sharing the MLX/Metal runtime with Gemma was the unlock: we get one
weight loader, one memory model, one cache to clear, and we can keep
Kokoro unloaded until the moment Gemma finishes a generation. See
[`03-memory-management.md`](03-memory-management.md) for the two-phase
serial unload that prevents the Kokoro/MLX cache-clear race.

For voice-input we picked the symmetric trade-off and stayed with
Apple: `SFSpeechRecognizer` is 0 MB of bundle weight, on-device by
default, and good enough at conversational English for short hiker
asks — bundling Whisper would have added ~500 MB for marginal gain.

## Streaming implementation

To make the audio feel responsive rather than batched, we plumbed
Kokoro for streaming synthesis: as Gemma 4 produces text, we cut the
stream into sentence-sized chunks and hand each chunk to Kokoro the
moment it's ready, then schedule playback back-to-back on the audio
queue. The hiker hears the first sentence in roughly 1–2 seconds
instead of waiting 5–10 seconds for the full answer to generate, and
subsequent sentences arrive continuously while the model is still
writing. The result is a conversation that feels like the companion is
thinking out loud, not reading from a script.

Three sharp edges we had to file down to make streaming sound natural,
all captured in the TTS section of
[`07-optimizations-and-future.md`](07-optimizations-and-future.md#4-tts--audio):

- **3-stage chunker** (sentence → clause → hard 80-char cap). Kokoro's
  duration predictor becomes unstable past ~80 characters, so very
  long sentences are split on clause boundaries before any hard cut to
  preserve prosody.
- **Text normalization** for dashes, smart quotes, and ellipses before
  G2P (grapheme-to-phoneme), so synthesis doesn't artifact on
  punctuation the predictor wasn't trained on.
- **Interrupt/resume with remainder**: a stop flag is checked mid-
  synthesis loop and any unspoken tail is preserved, so if the hiker
  taps to interrupt the companion mid-sentence we don't re-synthesize
  what they already heard.

For the full lifecycle (memory, audio-session switching between
record/playback, lazy loading) see
[`02-architecture-ios-app.md`](02-architecture-ios-app.md) and
[`03-memory-management.md`](03-memory-management.md).

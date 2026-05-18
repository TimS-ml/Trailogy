# iOS App Architecture (Trailogy)

## TL;DR

- Trailogy is an offline iOS trail guide that runs language, vision, speech, and retrieval models on the device with no runtime network dependency.
- SwiftUI views call long-lived services owned by the root view, and only the walking experience drives model inference.
- The app bundle includes the model files, trail content, retrieval corpus, and image assets needed for normal use.
- The public app name is Trailogy, while some internal target and source names remain unchanged to avoid provisioning churn.

## What The App Does

Trailogy is an offline AI companion for a guided hike. In the product model,
the trail package is available before the hike; in this repo/demo, the package
assets are bundled with the app. During the hike, the app can narrate stops,
answer spoken questions, answer photo-based questions, and generate a recap at
the end.

The hard requirement is no runtime network dependency. The app bundle contains
the models, trail content, retrieval corpus, and image assets needed for normal
use.

## Runtime Stack

| Component | Role |
|---|---|
| Gemma 4 E2B | Local language and vision reasoning. |
| MiniLM | Local embedding model for retrieval over bundled trail facts. |
| Kokoro | Local text-to-speech for natural spoken output. |
| Apple speech recognition | On-device speech-to-text for hold-to-ask. |
| MLX Swift / Metal | Shared runtime for Gemma, Kokoro, and embeddings. |

Gemma is not the only source of truth. Trail facts live in the preloaded trail
package and are retrieved into the prompt. Gemma turns those facts into a short,
spoken explanation.

## App Structure

```text
SwiftUI views
  |
  v
AppRouter, WalkingView, PickerView, DetailView, JournalView
  |
  v
GemmaService, RAGService, ValidationRunner, SpeechRecognizer, CameraController
  |
  v
MLX / Kokoro / MiniLM / SFSpeechRecognizer / AVFoundation
```

`ContentView` owns the shared services. `WalkingView` is the only view that
drives model inference; the other screens select trails, show details, cache
assets, or display the recap.

## User Flow

```text
Pick trail -> confirm package is available -> begin walk
    |
    v
At each stop: play narration about what is nearby
    |
    v
Between stops: prompt the user to look for a visible object
    |
    v
User may hold mic or take photo to ask a question
    |
    v
End hike: generate recap from what the user heard, asked, and saw
```

The interaction model is intentionally voice-first: hikers should keep their
eyes on the environment, not on a chat screen.

## Ask Pipeline

Text questions use retrieval:

```text
hold mic -> transcript -> retrieve trail chunks -> Gemma text answer -> Kokoro speech
```

Photo questions use the vision path:

```text
capture photo -> Gemma VLM answer with [camera=on] marker -> Kokoro speech
```

The image path does not currently inject RAG or long chat history. That keeps
the context small and makes the answer primarily about the photo. Text asks use
retrieval and short conversation history.

## Memory Discipline

The architecture is constrained by the fact that multiple MLX-backed models
share the same allocator. The important runtime rule is simple:

```text
Gemma loads -> Gemma generates -> Gemma unloads -> cache clears -> Kokoro speaks
```

Kokoro also unloads in a two-phase serial queue so MLX cache clearing happens
after local references are released. This avoids a subtle allocator race where
buffers return to the cache after the cache has already been cleared.

See [`03-memory-management.md`](03-memory-management.md) for the full memory
numbers and implementation details.

## Why The FSM Matters

The walking experience has two overlapping flows:

| Flow | Purpose |
|---|---|
| Tour phase FSM | Moves through `atStop -> between -> approaching -> atStop -> complete`. |
| Ask flow | Temporarily pauses narration while the user records, receives, or listens to an answer. |

This prevents narration, speech recognition, answer generation, and TTS from
fighting each other. It also supports the "look-for / payoff" learning arc: the
app asks the user to notice something first, then explains it at the next stop.

## Offline Package Contents

A trail package contains:

- stop narratives and local references;
- retrieval chunks and embeddings;
- images and voice assets;
- prompts and metadata needed for local grounding.

This is why Trailogy can answer place-specific questions without pretending the
model memorized every local fact.

## Cross-References

- Model pipeline: [`01-architecture-model-pipeline.md`](01-architecture-model-pipeline.md)
- Memory management: [`03-memory-management.md`](03-memory-management.md)
- RAG runtime: [`05-rag-runtime.md`](05-rag-runtime.md)
- Scene phase / Metal background handling: [`06-scenephase-metal-background.md`](06-scenephase-metal-background.md)

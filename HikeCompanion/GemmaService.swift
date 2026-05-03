// GemmaService.swift
// Wraps mlx-swift-lm's `ChatSession` over Gemma 4 E2B (INT4 quantized,
// ~3.5 GB). Loads from a bundled directory at `Bundle/Models/Gemma/`,
// exposes streaming + batch response APIs.
//
// Model files come from `mlx-community/gemma-4-e2b-it-4bit` (or unsloth's
// UD-MLX-4bit fallback) via `scripts/fetch-gemma.sh`.

import Foundation
import HuggingFace
import Hub
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

@MainActor
final class GemmaService: ObservableObject {

    // MARK: - Published state

    @Published private(set) var status: String = "Idle"
    @Published private(set) var isReady: Bool = false
    @Published private(set) var loadProgress: Double = 0

    // MARK: - Internals

    /// We hold the loaded model container for reuse across turns, but
    /// create a *fresh* `ChatSession` per question so KV cache doesn't
    /// accumulate. Multi-turn coherence is sacrificed in exchange for
    /// bounded resident memory — important on iPhone where Gemma + Kokoro
    /// already eat ~4 GB resident.
    private var modelContainer: ModelContainer?

    private let systemInstructions = """
    You are a friendly outdoor companion who helps hikers understand what they \
    see — geology, plants, animals, weather, and climate change. Keep responses \
    brief and conversational: 2 to 4 short sentences. Speak as if narrating, \
    not as if writing a report.
    """

    // MARK: - Lifecycle

    init() {
        Task { await loadAsync() }
    }

    private func loadAsync() async {
        // Models dir is included via xcodegen `type: folder` (no
        // `buildPhase: resources`) which preserves the directory tree.
        // Gemma's safetensors lives in its own subdirectory, isolated
        // from Kokoro's safetensors at Bundle/Models/kokoro-v1_0.safetensors
        // — important because mlx-swift-lm globs `*.safetensors` from the
        // directory we hand it, and we don't want it to load Kokoro's
        // weights into the Gemma model graph.
        let modelDir = Bundle.main.bundleURL
            .appendingPathComponent("Models")
            .appendingPathComponent("Gemma")
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("config.json").path
        ) else {
            status = "Gemma model missing — run scripts/fetch-gemma.sh, then bash scripts/generate-project.sh, then rebuild."
            return
        }

        status = "Loading Gemma 4 (10–30 s)…"
        do {
            modelContainer = try await loadModelContainer(
                from: modelDir,
                using: #huggingFaceTokenizerLoader()
            )
            isReady = true
            status = "Gemma 4 ready"
        } catch {
            status = "Load error: \(error.localizedDescription)"
        }
    }

    // MARK: - Inference

    /// Stream Gemma's response token-by-token. Returns nil if model not ready.
    /// Builds a *fresh* `ChatSession` per call so KV cache state doesn't
    /// accumulate across questions.
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error>? {
        guard let container = modelContainer else { return nil }
        let session = ChatSession(
            container,
            instructions: systemInstructions,
            generateParameters: GenerateParameters(temperature: 0.7)
        )
        return session.streamResponse(to: prompt)
    }
}

// GemmaService.swift
// Wraps mlx-swift-lm's `ChatSession` over Gemma 4 E2B (INT4 quantized,
// ~3.5 GB).
//
// LIFECYCLE:
//   • Lazy load on first Ask. App launches with only Kokoro resident.
//   • Stays loaded across Asks (faster turns — no 10–30 s reload).
//   • Conversation history is persisted in this service and replayed
//     into a fresh ChatSession every turn so Gemma can resolve "they",
//     "that", "do you remember" across multi-turn dialogue.
//   • Caller can `reset()` to wipe history without unloading the model.
//   • Caller can `unload()` to free the ~3 GB if they want memory back
//     (we don't auto-unload anymore).
//
// MEMORY (iPhone 17 Pro):
//   • Always-resident: ~3 GB Gemma weights (after first load).
//   • Peak during generation: +KV cache, scales with history length.
//   • Peak during Kokoro TTS post-generation: ~4 GB total. Still well
//     under the ~5 GB jetsam line.
//
// HISTORY (uncapped for now):
//   • Every (user, assistant) pair appended after stream completion.
//   • A long conversation will grow KV cache during generation since
//     we replay the whole history. Add a turn cap if it becomes an
//     issue. For typical hike-Q&A turns this should be fine.

import Foundation
import HuggingFace
import Hub
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

@MainActor
final class GemmaService: ObservableObject {

    // MARK: - Published state

    @Published private(set) var status: String = "Idle (Gemma loads on first Ask)"
    @Published private(set) var isLoaded: Bool = false
    @Published private(set) var historyTurnCount: Int = 0

    // MARK: - Internals

    private var modelContainer: ModelContainer?
    private var conversationHistory: [Chat.Message] = []

    private let systemInstructions = """
    You are a friendly outdoor companion who helps hikers understand what they \
    see — geology, plants, animals, weather, and climate change. Keep responses \
    brief and conversational: 2 to 4 short sentences. Speak as if narrating, \
    not as if writing a report. Remember earlier turns of this conversation \
    when answering follow-up questions.
    """

    // MARK: - Lifecycle

    /// Load the model into memory. Idempotent.
    func loadIfNeeded() async throws {
        guard modelContainer == nil else { return }

        let modelDir = Bundle.main.bundleURL
            .appendingPathComponent("Models")
            .appendingPathComponent("Gemma")
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("config.json").path
        ) else {
            throw GemmaError.modelMissing
        }

        status = "Loading Gemma 4 (10–30 s)…"
        modelContainer = try await loadModelContainer(
            from: modelDir,
            using: #huggingFaceTokenizerLoader()
        )
        isLoaded = true
        status = "Gemma 4 loaded"
    }

    /// Drop the model from memory. Use sparingly — reload costs 10–30 s.
    /// Conversation history is preserved across unload/reload.
    func unload() {
        modelContainer = nil
        isLoaded = false
        Memory.clearCache()
        status = "Gemma unloaded (history kept; next Ask will reload)"
    }

    /// Wipe the conversation history. Does not unload the model.
    func reset() {
        conversationHistory.removeAll()
        historyTurnCount = 0
        status = isLoaded ? "Gemma 4 loaded · history reset" : status
    }

    // MARK: - Inference

    /// Stream Gemma's response, with conversation history replayed so
    /// follow-ups can reference prior turns. Appends the (prompt, full
    /// response) pair to history after the stream completes.
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error>? {
        guard let container = modelContainer else { return nil }

        // Snapshot current history; ChatSession will consume it.
        let historySnapshot = conversationHistory

        let session = ChatSession(
            container,
            instructions: systemInstructions,
            history: historySnapshot,
            generateParameters: GenerateParameters(temperature: 0.7)
        )

        return AsyncThrowingStream { continuation in
            Task { @MainActor in
                var fullText = ""
                do {
                    for try await chunk in session.streamResponse(to: prompt) {
                        fullText += chunk
                        continuation.yield(chunk)
                    }
                    // Persist the turn to history.
                    self.conversationHistory.append(.init(role: .user, content: prompt))
                    self.conversationHistory.append(.init(role: .assistant, content: fullText))
                    self.historyTurnCount = self.conversationHistory.count / 2
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

enum GemmaError: LocalizedError {
    case modelMissing

    var errorDescription: String? {
        switch self {
        case .modelMissing:
            return "Gemma model missing — run scripts/fetch-gemma.sh, then bash scripts/generate-project.sh, then rebuild."
        }
    }
}

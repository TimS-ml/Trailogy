// ContentView.swift
// Root view. Owns the three @StateObject services
// (Gemma, Kokoro/ValidationRunner, SpeechRecognizer) and the AppRouter,
// and switches between the four main screens with a fade transition.
//
// All screen-specific UI lives in HikeCompanion/Views/*.swift.
// The original "develop everything in one file" UI (Form-based debug
// console with memory profiler, Speak-only TTS, voice/speed pickers,
// conversation reset) is preserved verbatim in `Views/DebugView.swift`
// and surfaced as a sheet from PickerView's bug icon.
//
// Mockup: design/mockups.html (Nature companion). The view-switch
// state machine is mirrored in `AppRouter`.

import SwiftUI

struct ContentView: View {
    @StateObject private var gemma = GemmaService()
    @StateObject private var tts = ValidationRunner()
    @StateObject private var speech = SpeechRecognizer()
    @StateObject private var rag = RAGService()
    @StateObject private var router = AppRouter()

    var body: some View {
        ZStack {
            switch router.screen {
            case .picker:
                PickerView()
                    .transition(.opacity)
            case .detail:
                DetailView()
                    .transition(.opacity)
            case .walking:
                WalkingView()
                    .transition(.opacity)
            case .journal:
                JournalView()
                    .transition(.opacity)
            }
        }
        .preferredColorScheme(.dark)
        .environmentObject(router)
        .environmentObject(gemma)
        .environmentObject(tts)
        .environmentObject(speech)
        .environmentObject(rag)
        // Preload the MiniLM embedder at launch in the background.
        // ~40 MB resident (small next to Gemma's 2.8 GB and Kokoro's
        // ~300 MB peak), and first-run downloads ~80 MB from HF — so
        // we start that before the user has a chance to ask. By the
        // time they hit the mic, the embedder is hot.
        .task {
            do {
                try await rag.preload()
            } catch {
                print("[RAG] preload failed: \(error.localizedDescription) — retrieval will retry on first use")
            }
        }
        .sheet(isPresented: $router.debugVisible) {
            DebugView()
                .environmentObject(gemma)
                .environmentObject(tts)
                .environmentObject(speech)
                .environmentObject(rag)
        }
    }
}

#Preview {
    ContentView()
}

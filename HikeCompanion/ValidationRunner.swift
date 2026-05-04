// ValidationRunner.swift
// Wraps mlalma/kokoro-ios (KokoroSwift / MLX). Loads model + voices from
// the app bundle, exposes one `synthesize(text:speed:)` call that times
// wall clock vs audio duration, plays the resulting audio, and saves a
// WAV to Documents/. Surfaces a live caption (currently-spoken word)
// using the per-token timestamps Kokoro returns.

import AVFoundation
import Foundation
import KokoroSwift
import MLX
import MLXUtilsLibrary

final class ValidationRunner: ObservableObject {

    // MARK: - Published state

    @Published private(set) var status: String = "Loading model…"
    @Published private(set) var voiceNames: [String] = []
    @Published var selectedVoice: String = ""
    @Published private(set) var lastResult: RunResult?
    @Published private(set) var lastWavURL: URL?
    @Published private(set) var isReady: Bool = false
    /// True from `synthesize` start until the synth+play workQueue closure
    /// returns. Note: this can be `false` while audio is still playing
    /// (synth completes before playback finishes).
    @Published private(set) var isRunning: Bool = false
    /// True while audio is actively playing through the player node.
    /// Use this (not `isRunning`) to know whether the user is hearing
    /// something. Goes true when playback starts, false when the caption
    /// timer reaches the last token (or `stop()` is called).
    @Published private(set) var isSpeaking: Bool = false
    /// Words spoken so far during current playback. Updates ~every 50 ms via Timer.
    @Published private(set) var currentCaption: String = ""

    /// When true, the next `play(...)` call bails out and any in-flight
    /// synth result is discarded. Cleared on the next `synthesize(...)` entry.
    private var stopRequested: Bool = false

    // MARK: - Internals

    /// KokoroTTS instance. Lazy-loaded on first synth (and re-loaded on
    /// each subsequent synth) so the ~310 MB of MLX-resident model
    /// weights aren't sitting around when we're not actively speaking.
    /// Kokoro is non-autoregressive (single forward pass per chunk),
    /// so there's no rolling state to preserve between synths.
    private var tts: KokoroTTS?
    /// Path to the safetensors file in the bundle. Kept so we can
    /// re-init `KokoroTTS` after each unload without a Bundle lookup.
    private var modelURL: URL?
    /// Voice embeddings from voices.npz. Tiny (~14 MB), kept resident.
    private var voices: [String: MLXArray] = [:]
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?

    // Caption timing
    private var captionTokens: [CaptionToken] = []
    private var captionTimer: Timer?
    private var captionStartTime: Date?

    private let workQueue = DispatchQueue(label: "com.lijuncheng16.HikeCompanion.runner",
                                          qos: .userInitiated)

    // MARK: - Lifecycle

    init() {
        loadAsync()
    }

    private func loadAsync() {
        workQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.loadSync()
                DispatchQueue.main.async {
                    self.isReady = true
                    self.status = "Idle"
                }
            } catch {
                DispatchQueue.main.async {
                    self.status = "Load error: \(error.localizedDescription)"
                }
            }
        }
    }

    private func loadSync() throws {
        // Files live in the `Models/` blue-folder reference (xcodegen
        // `type: folder`), not at the bundle root. `Bundle.url(forResource:
        // withExtension:)` only searches the root unless you pass
        // `subdirectory:`.
        guard let modelURL = Bundle.main.url(forResource: "kokoro-v1_0",
                                             withExtension: "safetensors",
                                             subdirectory: "Models") else {
            throw RunnerError.modelMissing
        }
        guard let voicesURL = Bundle.main.url(forResource: "voices",
                                              withExtension: "npz",
                                              subdirectory: "Models") else {
            throw RunnerError.voicesMissing
        }
        self.modelURL = modelURL

        // Voices are tiny (~14 MB) and we need their names for the
        // picker even when Kokoro isn't loaded — keep them resident.
        DispatchQueue.main.async { self.status = "Loading voices…" }
        let loadedVoices = NpyzReader.read(fileFromPath: voicesURL) ?? [:]
        self.voices = loadedVoices

        let names = loadedVoices.keys
            .map { String($0.split(separator: ".")[0]) }
            .sorted()
        DispatchQueue.main.async {
            self.voiceNames = names
            self.selectedVoice = names.first(where: { $0 == "af_bella" }) ?? names.first ?? ""
        }

        // Audio engine for playback (keep alive across synths — owns the
        // PCM buffer queue, independent of the Kokoro model).
        let aEngine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        aEngine.attach(player)
        self.audioEngine = aEngine
        self.playerNode = player

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
        } catch {
            // Non-fatal.
        }

        // KokoroTTS itself is lazy-loaded on first `synthesize`; see
        // `ensureModelLoaded()`.
    }

    /// Initialise (or re-initialise) the KokoroTTS engine. Idempotent
    /// — no-op if already loaded. Called at the start of each synth.
    private func ensureModelLoaded() throws {
        guard tts == nil else { return }
        guard let modelURL else { throw RunnerError.modelMissing }
        tts = KokoroTTS(modelPath: modelURL)
    }

    /// Drop the KokoroTTS engine reference + clear MLX cache. Audio
    /// already queued on `AVAudioPlayerNode` plays independently — that
    /// buffer lives in AVFoundation memory, not MLX.
    private func unloadModel() {
        tts = nil
        Memory.clearCache()
    }

    // MARK: - Synthesize

    func synthesize(text: String, speed: Float = 1.0) {
        guard isReady, !isRunning else { return }
        let voiceKey = selectedVoice
        guard let voiceArray = voices["\(voiceKey).npy"] else {
            DispatchQueue.main.async {
                self.status = "No voice array for \(voiceKey)"
            }
            return
        }

        DispatchQueue.main.async {
            self.isRunning = true
            self.stopRequested = false
            self.status = "Loading Kokoro…"
            self.currentCaption = ""
            self.stopCaptionTimer()
            MemoryStats.log("kokoro.synthesize start")
        }

        workQueue.async { [weak self] in
            guard let self else { return }
            do {
                // Lazy-load Kokoro for this synth. Adds ~1–2 s on the
                // first chunk; reused across chunks within the same call.
                try self.ensureModelLoaded()
                guard let tts = self.tts else { throw RunnerError.modelMissing }

                DispatchQueue.main.async {
                    self.status = "Synthesising…"
                    MemoryStats.log("kokoro.model loaded")
                }
                let language: Language = (voiceKey.first == "a") ? .enUS : .enGB
                let chunks = Self.splitForSynthesis(text)

                var combined: [Float] = []
                var allCaptionTokens: [CaptionToken] = []
                var chunkOffsetSec: Double = 0
                let sampleRate = Double(KokoroTTS.Constants.samplingRate)

                let wallStart = Date().timeIntervalSince1970

                for (i, chunk) in chunks.enumerated() {
                    DispatchQueue.main.async {
                        self.status = "Synthesising chunk \(i+1)/\(chunks.count)…"
                    }

                    let (audio, mtokens) = try tts.generateAudio(
                        voice: voiceArray,
                        language: language,
                        text: chunk,
                        speed: speed
                    )

                    // Per-chunk MTokens → global-time CaptionTokens
                    if let toks = mtokens {
                        for t in toks {
                            guard let s = t.start_ts, let e = t.end_ts else { continue }
                            allCaptionTokens.append(CaptionToken(
                                text: t.text,
                                whitespace: t.whitespace,
                                startTs: chunkOffsetSec + s,
                                endTs: chunkOffsetSec + e
                            ))
                        }
                    }

                    let chunkAudioDur = Double(audio.count) / sampleRate
                    combined.append(contentsOf: audio)

                    // 50 ms silence between chunks (click suppression);
                    // bump the offset so subsequent token timestamps stay correct.
                    if i < chunks.count - 1 {
                        let silenceFrames = Int(sampleRate * 0.05)
                        combined.append(contentsOf: [Float](repeating: 0, count: silenceFrames))
                        chunkOffsetSec += chunkAudioDur + 0.05
                    } else {
                        chunkOffsetSec += chunkAudioDur
                    }
                }

                let wallTime = Date().timeIntervalSince1970 - wallStart
                let audioDur = Double(combined.count) / sampleRate
                let rtf = audioDur > 0 ? (wallTime / audioDur) : 0

                let wavURL = try Self.saveWav(audio: combined, sampleRate: sampleRate)

                let result = RunResult(
                    text: text,
                    voice: voiceKey,
                    speed: speed,
                    wallTimeSec: wallTime,
                    audioDurationSec: audioDur,
                    rtf: rtf,
                    wavURL: wavURL,
                    chunkCount: chunks.count
                )

                DispatchQueue.main.async {
                    self.lastResult = result
                    self.lastWavURL = wavURL
                    let chunkSuffix = chunks.count > 1 ? "  (\(chunks.count) chunks)" : ""
                    self.status = String(
                        format: "RTF %.3f  (%.1f× realtime)  audio %.2f s  wall %.2f s%@",
                        rtf, rtf > 0 ? 1.0 / rtf : 0, audioDur, wallTime, chunkSuffix
                    )
                    self.isRunning = false
                    MemoryStats.log("kokoro.synthesize done")
                }

                self.captionTokens = allCaptionTokens
                self.play(audio: combined, sampleRate: sampleRate)
                // End of phase 1. Local `tts` binding will be released
                // when this closure exits. Phase 2 (below) is queued on
                // the same SERIAL workQueue and runs only after this
                // closure has fully exited — by then the only remaining
                // strong ref to KokoroTTS is `self.tts`, which phase 2
                // will nil out. Critical for MLX cache to actually drop
                // the model weights instead of holding them as cached
                // buffers we never clear.
            } catch {
                DispatchQueue.main.async {
                    self.status = "Synth error: \(error.localizedDescription)"
                    self.isRunning = false
                }
            }
        }

        // Phase 2: drop the KokoroTTS engine and clear MLX's cache.
        // This block runs AFTER phase 1's closure has exited (serial
        // queue), so the local strong reference inside that closure is
        // already gone. Setting `self.tts = nil` now is the last drop;
        // ARC deinits KokoroTTS, MLX returns its buffers into the cache
        // pool, and `Memory.clearCache()` immediately releases them.
        workQueue.async { [weak self] in
            guard let self else { return }
            self.tts = nil
            Memory.clearCache()
            DispatchQueue.main.async {
                // Status reflects truth: model is unloaded again.
                // The just-completed synth's RTF is preserved in
                // `lastResult` for the "Last TTS run" section.
                self.status = "Idle"
                MemoryStats.log("kokoro.unload done")
            }
        }
    }

    // MARK: - Playback + caption timer

    private func play(audio: [Float], sampleRate: Double) {
        // If the user interrupted (e.g. held the mic to ask a question)
        // between synth completion and this point, drop the audio rather
        // than blasting it over their question.
        if stopRequested { return }
        guard let engine = audioEngine, let player = playerNode else { return }
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate,
                                         channels: 1) else { return }
        guard let buf = AVAudioPCMBuffer(pcmFormat: format,
                                         frameCapacity: AVAudioFrameCount(audio.count)) else { return }
        buf.frameLength = buf.frameCapacity
        let dst = buf.floatChannelData![0]
        audio.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            dst.update(from: base, count: src.count)
        }
        engine.connect(player, to: engine.mainMixerNode, format: format)
        do {
            if !engine.isRunning { try engine.start() }
        } catch { return }
        player.scheduleBuffer(buf, at: nil, options: .interrupts, completionHandler: nil)
        if !player.isPlaying { player.play() }

        // Kick off caption sync on main run loop.
        DispatchQueue.main.async { [weak self] in
            self?.startCaptionTimer()
        }
    }

    func playLastAgain() {
        guard let url = lastWavURL else { return }
        do {
            let file = try AVAudioFile(forReading: url)
            guard let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                             frameCapacity: AVAudioFrameCount(file.length)) else { return }
            try file.read(into: buf)
            guard let engine = audioEngine, let player = playerNode else { return }
            engine.connect(player, to: engine.mainMixerNode, format: file.processingFormat)
            if !engine.isRunning { try engine.start() }
            player.scheduleBuffer(buf, at: nil, options: .interrupts, completionHandler: nil)
            if !player.isPlaying { player.play() }
            startCaptionTimer()
        } catch {
            // ignore
        }
    }

    /// Walks captionTokens in order, advancing as audioTime crosses each
    /// token's start_ts. Updates currentCaption on the main run loop.
    /// Also drives `isSpeaking` — true while audio is reaching the user.
    private func startCaptionTimer() {
        stopCaptionTimer()
        guard !captionTokens.isEmpty else { return }
        currentCaption = ""
        isSpeaking = true
        captionStartTime = Date()
        var nextIndex = 0

        captionTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] timer in
            guard let self, let start = self.captionStartTime else {
                timer.invalidate(); return
            }
            let elapsed = Date().timeIntervalSince(start)

            // Append every token whose start has been reached.
            while nextIndex < self.captionTokens.count,
                  self.captionTokens[nextIndex].startTs <= elapsed {
                let t = self.captionTokens[nextIndex]
                let sep = t.whitespace.isEmpty ? "" : t.whitespace
                self.currentCaption += t.text + sep
                nextIndex += 1
            }

            // Done when we've passed the last token's end + a small grace period.
            if let last = self.captionTokens.last, elapsed > last.endTs + 0.5 {
                timer.invalidate()
                self.captionTimer = nil
                self.isSpeaking = false
            }
        }
    }

    private func stopCaptionTimer() {
        captionTimer?.invalidate()
        captionTimer = nil
        captionStartTime = nil
    }

    // MARK: - Stop / interrupt

    /// Halt audio playback immediately. Used when the user holds the mic
    /// to interrupt an ongoing tour-guide narration. Safe to call from
    /// any thread; bounces work to main where needed.
    ///
    /// Side effects:
    ///   • `playerNode.stop()` halts the audio buffer (AVAudioPlayerNode)
    ///   • `stopRequested = true` so any synth still in flight discards
    ///     its output instead of restarting playback
    ///   • caption timer stops and `isSpeaking` flips to false
    ///   • **`isRunning` flips to false** so the next `synthesize(...)`
    ///     call doesn't silently bail on the `!isRunning` guard. This is
    ///     critical for the pause/resume path — without it, an in-flight
    ///     synth's main.async (which sets isRunning=false on completion)
    ///     might not have run yet by the time the user clicks resume,
    ///     and the resume's synthesize() returns immediately with no
    ///     audio.
    ///   • `currentCaption` is preserved at its current value so the
    ///     caller can read "what we got through" before resetting it
    func stop() {
        DispatchQueue.main.async {
            self.stopRequested = true
            self.playerNode?.stop()
            self.captionTimer?.invalidate()
            self.captionTimer = nil
            self.captionStartTime = nil
            self.isSpeaking = false
            self.isRunning = false
            self.status = "Idle (stopped)"
        }
    }

    // MARK: - Text chunking (avoid 510-token cap)

    // MARK: - Text chunking (phrase-level, to avoid Kokoro's start-glitch)

    /// Empirically: Kokoro's duration predictor goes unstable on inputs
    /// over ~60 characters and produces a high-pitch beep covering the
    /// first ~1–2 seconds of audio. The token-cap (510) is irrelevant
    /// at this scale — the bug fires far below it. So we chunk much
    /// more aggressively than the cap requires, on every phrase break,
    /// to keep each chunk safely below the trigger.
    private static let phraseDelimiters: [Character] = [".", "!", "?", ",", ":", ";"]

    /// Hard ceiling per chunk. Empirical safe zone is ~60 chars, but a
    /// run-on phrase between commas can be longer than that and still
    /// behave fine. 80 is a compromise — we hard-chop at word boundaries
    /// only past this point.
    private static let maxCharsPerChunk = 80

    private static func splitForSynthesis(_ text: String) -> [String] {
        let primary = splitOnDelimiters(text, delimiters: phraseDelimiters)
        var safe: [String] = []
        for s in primary {
            if s.count <= maxCharsPerChunk {
                safe.append(s)
            } else {
                safe.append(contentsOf: hardChop(s, maxChars: maxCharsPerChunk))
            }
        }
        let cleaned = safe
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return cleaned.isEmpty ? [text] : cleaned
    }

    private static func splitOnDelimiters(_ text: String, delimiters: [Character]) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if delimiters.contains(ch) {
                out.append(current)
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out
    }

    private static func hardChop(_ text: String, maxChars: Int) -> [String] {
        let words = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        var out: [String] = []
        var current = ""
        for w in words {
            if current.count + w.count + 1 > maxChars {
                if !current.isEmpty { out.append(current) }
                current = w
            } else {
                current = current.isEmpty ? w : current + " " + w
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    // MARK: - WAV save

    private static func saveWav(audio: [Float], sampleRate: Double) throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let ts = Int(Date().timeIntervalSince1970)
        let url = docs.appendingPathComponent("hike_\(ts).wav")
        try AudioUtils.writeWavFile(samples: audio, sampleRate: sampleRate, fileURL: url)
        return url
    }
}

// MARK: - Caption token (chunk-offsets pre-applied)

private struct CaptionToken {
    let text: String
    let whitespace: String
    /// Seconds from start of full concatenated audio.
    let startTs: Double
    let endTs: Double
}

// MARK: - Result type

struct RunResult: Identifiable {
    let id = UUID()
    let text: String
    let voice: String
    let speed: Float
    let wallTimeSec: Double
    let audioDurationSec: Double
    /// wallTime / audioDuration; lower is better (< 1.0 = faster than realtime).
    let rtf: Double
    let wavURL: URL
    let chunkCount: Int
}

// MARK: - Errors

enum RunnerError: LocalizedError {
    case modelMissing
    case voicesMissing

    var errorDescription: String? {
        switch self {
        case .modelMissing:
            return "kokoro-v1_0.safetensors not in app bundle. Run scripts/fetch-models.sh and rebuild."
        case .voicesMissing:
            return "voices.npz not in app bundle. Run scripts/fetch-models.sh and rebuild."
        }
    }
}

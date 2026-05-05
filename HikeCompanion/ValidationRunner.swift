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

    /// Synthesize and play `text` through Kokoro.
    ///
    /// **Streaming** — each chunk's audio buffer is scheduled on the
    /// player as soon as that chunk finishes synthesizing, instead of
    /// waiting for ALL chunks to be synthesized first. For a long
    /// intro paragraph this drops perceived start latency from ~5–10 s
    /// (synth-all-then-play) to ~1–2 s (synth-first-then-play, queue
    /// the rest as they arrive).
    ///
    /// Per-chunk caption tokens are appended to `captionTokens` on
    /// the main queue with their global-time offsets pre-applied, so
    /// the live caption (`currentCaption`) keeps growing in sync with
    /// audio across chunk boundaries.
    func synthesize(text: String, speed: Float = 1.0) {
        guard isReady, !isRunning else { return }
        let voiceKey = selectedVoice
        guard let voiceArray = voices["\(voiceKey).npy"] else {
            DispatchQueue.main.async {
                self.status = "No voice array for \(voiceKey)"
            }
            return
        }

        // Normalize problematic punctuation BEFORE chunking so dashes
        // / smart quotes / ellipses don't reach the G2P layer.
        let normalized = Self.normalizeForSynthesis(text)
        let chunks = Self.splitForSynthesis(normalized)
        guard !chunks.isEmpty else { return }

        // Synchronous when called from main (the normal path) so that
        // `isRunning = true` is observable IMMEDIATELY after this call
        // returns. Async fallback for the rare off-main caller.
        //
        // This matters because callers like `runAsk` do:
        //     tts.synthesize(text: fullText, speed: 1.0)
        //     await waitForTTSPlaybackToFinish()
        // both on MainActor, with no main-queue yield in between. If
        // we wrapped this prep block in `DispatchQueue.main.async`, the
        // wait would observe stale `isRunning == false` and return
        // instantly — clearing the on-screen answer text before Kokoro
        // had a chance to start speaking.
        let prep: () -> Void = {
            self.isRunning = true
            self.stopRequested = false
            self.status = "Loading Kokoro…"
            self.currentCaption = ""
            self.captionTokens = []
            self.captionStartTime = nil
            self.stopCaptionTimer()
            MemoryStats.log("kokoro.synthesize start")
        }
        if Thread.isMainThread {
            prep()
        } else {
            DispatchQueue.main.async(execute: prep)
        }

        workQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.ensureModelLoaded()
                guard let tts = self.tts else { throw RunnerError.modelMissing }

                DispatchQueue.main.async {
                    self.status = "Synthesising…"
                    MemoryStats.log("kokoro.model loaded")
                }
                let language: Language = (voiceKey.first == "a") ? .enUS : .enGB
                let sampleRate = Double(KokoroTTS.Constants.samplingRate)

                guard let engine = self.audioEngine,
                      let player = self.playerNode,
                      let format = AVAudioFormat(
                        standardFormatWithSampleRate: sampleRate, channels: 1)
                else {
                    DispatchQueue.main.async {
                        self.status = "Audio engine not initialised"
                        self.isRunning = false
                    }
                    return
                }

                // Reset any prior playback / scheduled buffers; reconnect
                // the player to the mixer in case format changed.
                engine.connect(player, to: engine.mainMixerNode, format: format)
                if !engine.isRunning {
                    try engine.start()
                }
                player.stop()

                let wallStart = Date().timeIntervalSince1970
                var combined: [Float] = []          // accumulated for saveWav
                var chunkOffsetSec: Double = 0
                var firstChunkScheduled = false
                let totalChunks = chunks.count

                for (i, chunk) in chunks.enumerated() {
                    if self.stopRequested { break }

                    DispatchQueue.main.async {
                        self.status = "Synthesising chunk \(i+1)/\(totalChunks)…"
                    }

                    let (audio, mtokens) = try tts.generateAudio(
                        voice: voiceArray,
                        language: language,
                        text: chunk,
                        speed: speed
                    )

                    if self.stopRequested { break }

                    let chunkAudioDur = Double(audio.count) / sampleRate

                    // Append this chunk's caption tokens with global
                    // timestamps to the main-thread captionTokens.
                    let myOffset = chunkOffsetSec
                    if let toks = mtokens {
                        let captionAddition: [CaptionToken] = toks.compactMap { t in
                            guard let s = t.start_ts, let e = t.end_ts else { return nil }
                            return CaptionToken(
                                text: t.text,
                                whitespace: t.whitespace,
                                startTs: myOffset + s,
                                endTs:   myOffset + e
                            )
                        }
                        DispatchQueue.main.async {
                            self.captionTokens.append(contentsOf: captionAddition)
                        }
                    }

                    // Build a PCM buffer for this chunk and schedule it.
                    // Note: NO `.interrupts` — we want chunks to queue
                    // back-to-back, not replace each other.
                    guard let buf = AVAudioPCMBuffer(
                        pcmFormat: format,
                        frameCapacity: AVAudioFrameCount(audio.count))
                    else {
                        chunkOffsetSec += chunkAudioDur
                        continue
                    }
                    buf.frameLength = buf.frameCapacity
                    if let dst = buf.floatChannelData?[0] {
                        audio.withUnsafeBufferPointer { src in
                            if let base = src.baseAddress {
                                dst.update(from: base, count: src.count)
                            }
                        }
                    }

                    // The completion handler on the LAST buffer flips
                    // isSpeaking false so the lyric area knows the lane
                    // ended. Earlier chunks have no completion handler.
                    let isLast = (i == totalChunks - 1)
                    if isLast {
                        player.scheduleBuffer(
                            buf,
                            at: nil,
                            options: [],
                            completionCallbackType: .dataPlayedBack
                        ) { [weak self] _ in
                            DispatchQueue.main.async {
                                self?.isSpeaking = false
                            }
                        }
                    } else {
                        player.scheduleBuffer(buf, at: nil, options: [],
                                              completionHandler: nil)
                    }

                    if !firstChunkScheduled {
                        firstChunkScheduled = true
                        DispatchQueue.main.async {
                            self.isSpeaking = true
                            // startCaptionTimer sets captionStartTime
                            // itself — see its docs.
                            self.startCaptionTimer()
                        }
                        if !player.isPlaying { player.play() }
                    }

                    combined.append(contentsOf: audio)
                    chunkOffsetSec += chunkAudioDur
                }

                // After all chunks scheduled, write the WAV (best-effort
                // for the debug-screen "Replay" feature) and update
                // status. Playback may continue from the queued buffers.
                let wallTime = Date().timeIntervalSince1970 - wallStart
                let audioDur = chunkOffsetSec
                let rtf = audioDur > 0 ? (wallTime / audioDur) : 0
                let wavURL = (try? Self.saveWav(audio: combined, sampleRate: sampleRate))

                DispatchQueue.main.async {
                    if let wavURL { self.lastWavURL = wavURL }
                    self.lastResult = RunResult(
                        text: text,
                        voice: voiceKey,
                        speed: speed,
                        wallTimeSec: wallTime,
                        audioDurationSec: audioDur,
                        rtf: rtf,
                        wavURL: wavURL ?? URL(fileURLWithPath: "/dev/null"),
                        chunkCount: totalChunks
                    )
                    self.status = String(
                        format: "RTF %.3f  audio %.2f s  wall %.2f s  (%d chunks)",
                        rtf, audioDur, wallTime, totalChunks
                    )
                    self.isRunning = false
                    MemoryStats.log("kokoro.synthesize done")
                }
            } catch {
                DispatchQueue.main.async {
                    self.status = "Synth error: \(error.localizedDescription)"
                    self.isRunning = false
                    // CRITICAL: also flip isSpeaking false on the error
                    // path. If we already started playing earlier chunks
                    // before the failing one, isSpeaking is currently
                    // true and the dataPlayedBack callback will NEVER
                    // fire (it was attached to the last chunk we never
                    // got to schedule). Without this line, anyone
                    // awaiting playback to end (WalkingView's runAsk)
                    // would hang indefinitely.
                    self.isSpeaking = false
                    self.captionTimer?.invalidate()
                    self.captionTimer = nil
                }
            }
        }

        // Phase 2: drop the KokoroTTS engine and clear MLX's cache.
        // Runs after phase 1 closure exits (serial workQueue). Phase 1
        // exits when ALL chunks have been scheduled; at that point the
        // KokoroTTS instance is no longer needed (the player has the
        // PCM buffers). Phase 2 nil's it out and clears MLX cache.
        workQueue.async { [weak self] in
            guard let self else { return }
            self.tts = nil
            Memory.clearCache()
            DispatchQueue.main.async {
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
    ///
    /// **Streaming-aware**: captionTokens grows over time as new chunks
    /// finish synthesizing (each chunk's tokens are appended on the
    /// main queue with global timestamps already applied). The timer
    /// keeps running and pulling tokens until `isSpeaking` flips false
    /// — that happens when the LAST scheduled buffer's playback
    /// completion callback fires (set up in `synthesize`). We don't
    /// use `captionTokens.last.endTs` as the stop signal because it
    /// keeps moving as more chunks arrive.
    ///
    /// Sets `captionStartTime = Date()` itself — must happen AFTER the
    /// `stopCaptionTimer()` call below, which nils it out. (Earlier
    /// versions set this from the call site, which broke the moment
    /// `stopCaptionTimer()` ran inside this function and erased the
    /// just-set timestamp; the timer then fired but bailed every tick
    /// on the `guard let start` check.)
    private func startCaptionTimer() {
        stopCaptionTimer()
        captionStartTime = Date()
        var nextIndex = 0

        captionTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] timer in
            guard let self else {
                timer.invalidate(); return
            }
            // The dataPlayedBack callback on the last buffer flips
            // isSpeaking false; that's our signal to retire.
            guard self.isSpeaking else {
                timer.invalidate()
                self.captionTimer = nil
                return
            }
            guard let start = self.captionStartTime else { return }
            let elapsed = Date().timeIntervalSince(start)

            while nextIndex < self.captionTokens.count,
                  self.captionTokens[nextIndex].startTs <= elapsed {
                let t = self.captionTokens[nextIndex]
                let sep = t.whitespace.isEmpty ? "" : t.whitespace
                self.currentCaption += t.text + sep
                nextIndex += 1
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

    // MARK: - Text normalization + chunking

    /// Replace punctuation Kokoro's grapheme-to-phoneme mishandles
    /// (em / en dashes, smart quotes, ellipses, compound hyphens) with
    /// ASCII equivalents the model was trained on. Without this we get
    /// audible artifacts at every dash and curly quote: a mechanical
    /// clip, an unnatural pause, or a mispronunciation of the
    /// surrounding word.
    ///
    /// Specifically:
    ///   • " — "  → ", "        (em dash with spaces = parenthetical pause)
    ///   • "—"     → " "         (tight em dash = soft join)
    ///   • smart quotes → ASCII apostrophe / quote
    ///   • "…"     → ". "
    ///   • "word-word" → "word word"  (compound hyphen becomes a soft
    ///     word break — see comment below)
    /// Then collapses double spaces that the substitutions can leave behind.
    ///
    /// **Compound hyphens.** Trail narration is full of compound words
    /// like "year-round", "Howe-truss", "four-story", "eighty-ton",
    /// "moss-covered". Misaki's G2P treats `word-word` as TWO prosodic
    /// units with a hard boundary at the hyphen, which sounds like a
    /// glitch / sharp clip / unnatural cut between the halves. Replacing
    /// the hyphen with a space turns each side into a normal word; the
    /// resulting prosody has a natural soft join instead of an artifact.
    /// We restrict the replacement to hyphens flanked by word characters
    /// on BOTH sides so leading/trailing hyphens (rare in narration) are
    /// left alone.
    private static func normalizeForSynthesis(_ text: String) -> String {
        var s = text
        s = s.replacingOccurrences(of: " — ", with: ", ")
        s = s.replacingOccurrences(of: " – ", with: ", ")
        s = s.replacingOccurrences(of: "—",   with: " ")
        s = s.replacingOccurrences(of: "–",   with: " ")
        s = s.replacingOccurrences(of: "\u{2018}", with: "'")  // '
        s = s.replacingOccurrences(of: "\u{2019}", with: "'")  // '
        s = s.replacingOccurrences(of: "\u{201C}", with: "\"") // "
        s = s.replacingOccurrences(of: "\u{201D}", with: "\"") // "
        s = s.replacingOccurrences(of: "\u{2026}", with: ". ")
        // Compound hyphens: only between word characters on both sides.
        s = s.replacingOccurrences(
            of: "(?<=\\w)-(?=\\w)",
            with: " ",
            options: .regularExpression
        )
        while s.contains("  ") {
            s = s.replacingOccurrences(of: "  ", with: " ")
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Why we chunk: Kokoro's duration predictor goes unstable on
    /// inputs over ~60 characters and produces a high-pitch beep
    /// covering the first ~1–2 s of audio. So we keep each chunk
    /// below ~80 chars.
    ///
    /// Why we PREFER sentence boundaries: each chunk is synthesized
    /// independently, so the model can't carry prosody across the
    /// boundary. Splitting at a comma mid-clause makes the resulting
    /// audio sound mechanical (a sharp beat where there should be a
    /// flowing intonation curve). Splitting at a `. ! ?` is natural —
    /// prosody resets there anyway.
    ///
    /// Strategy: split on sentence enders first. Only fall back to
    /// clause delimiters (`, : ;`) for sentences that exceed the
    /// 80-char ceiling. Hard-chop at word boundaries as last resort.
    private static let sentenceEnders: [Character] = [".", "!", "?"]
    private static let clauseDelimiters: [Character] = [",", ":", ";"]
    private static let maxCharsPerChunk = 80

    private static func splitForSynthesis(_ text: String) -> [String] {
        // Stage 1: split on sentence enders.
        let stage1 = splitOnDelimiters(text, delimiters: sentenceEnders)
        // Stage 2: any sentence still > maxCharsPerChunk → split on clauses.
        var stage2: [String] = []
        for s in stage1 {
            if s.count <= maxCharsPerChunk {
                stage2.append(s)
            } else {
                stage2.append(contentsOf:
                    splitOnDelimiters(s, delimiters: clauseDelimiters))
            }
        }
        // Stage 3: anything STILL too long → hard-chop at word boundaries.
        var stage3: [String] = []
        for s in stage2 {
            if s.count <= maxCharsPerChunk {
                stage3.append(s)
            } else {
                stage3.append(contentsOf: hardChop(s, maxChars: maxCharsPerChunk))
            }
        }
        let cleaned = stage3
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

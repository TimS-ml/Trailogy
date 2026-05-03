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
    @Published private(set) var isRunning: Bool = false
    /// Words spoken so far during current playback. Updates ~every 50 ms via Timer.
    @Published private(set) var currentCaption: String = ""

    // MARK: - Internals

    private var tts: KokoroTTS?
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
                    self.status = "Ready. Type something and tap Synthesize."
                }
            } catch {
                DispatchQueue.main.async {
                    self.status = "Load error: \(error.localizedDescription)"
                }
            }
        }
    }

    private func loadSync() throws {
        guard let modelURL = Bundle.main.url(forResource: "kokoro-v1_0",
                                             withExtension: "safetensors") else {
            throw RunnerError.modelMissing
        }
        guard let voicesURL = Bundle.main.url(forResource: "voices",
                                              withExtension: "npz") else {
            throw RunnerError.voicesMissing
        }

        DispatchQueue.main.async { self.status = "Initialising KokoroTTS (~10–30 s)…" }
        let engine = KokoroTTS(modelPath: modelURL)
        self.tts = engine

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

        // Audio engine for playback.
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
    }

    // MARK: - Synthesize

    func synthesize(text: String, speed: Float = 1.0) {
        guard isReady, !isRunning, let tts = tts else { return }
        let voiceKey = selectedVoice
        guard let voiceArray = voices["\(voiceKey).npy"] else {
            DispatchQueue.main.async {
                self.status = "No voice array for \(voiceKey)"
            }
            return
        }

        DispatchQueue.main.async {
            self.isRunning = true
            self.status = "Synthesising…"
            self.currentCaption = ""
            self.stopCaptionTimer()
        }

        workQueue.async { [weak self] in
            guard let self else { return }
            do {
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

                    // Convert per-chunk MTokens to global-time CaptionTokens
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

                    // 50 ms silence between chunks (already used for click suppression);
                    // also bump the offset so subsequent token timestamps stay correct.
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
                }

                self.captionTokens = allCaptionTokens
                self.play(audio: combined, sampleRate: sampleRate)
            } catch {
                DispatchQueue.main.async {
                    self.status = "Synth error: \(error.localizedDescription)"
                    self.isRunning = false
                }
            }
        }
    }

    // MARK: - Playback + caption timer

    private func play(audio: [Float], sampleRate: Double) {
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
    private func startCaptionTimer() {
        stopCaptionTimer()
        guard !captionTokens.isEmpty else { return }
        currentCaption = ""
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
            }
        }
    }

    private func stopCaptionTimer() {
        captionTimer?.invalidate()
        captionTimer = nil
        captionStartTime = nil
    }

    // MARK: - Text chunking (avoid 510-token cap)

    private static func splitForSynthesis(_ text: String) -> [String] {
        let primary = splitOnDelimiters(text, delimiters: [".", "!", "?"])
        var safe: [String] = []
        let maxChars = 250
        for s in primary {
            if s.count <= maxChars {
                safe.append(s)
            } else {
                let secondary = splitOnDelimiters(s, delimiters: [",", ";"])
                if secondary.allSatisfy({ $0.count <= maxChars }) {
                    safe.append(contentsOf: secondary)
                } else {
                    safe.append(contentsOf: hardChop(s, maxChars: maxChars))
                }
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

// ValidationRunner.swift
// Wraps mlalma/kokoro-ios (KokoroSwift / MLX). Loads model + voices from
// the app bundle, exposes one `synthesize(text:)` call that times wall
// clock vs audio duration, plays the resulting audio, and saves a WAV
// to Documents/.

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

    // MARK: - Internals

    private var tts: KokoroTTS?
    private var voices: [String: MLXArray] = [:]
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?

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

        // Voice keys in voices.npz look like "af_bella.npy". Strip the suffix
        // for display, sort alphabetically.
        let names = loadedVoices.keys
            .map { String($0.split(separator: ".")[0]) }
            .sorted()
        DispatchQueue.main.async {
            self.voiceNames = names
            // Default to af_bella if present, else first.
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
            // Non-fatal; playback may still work.
        }
    }

    // MARK: - Synthesize

    func synthesize(text: String) {
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
        }

        workQueue.async { [weak self] in
            guard let self else { return }
            do {
                // Voice name convention: prefix 'a' = US English, else GB.
                let language: Language = (voiceKey.first == "a") ? .enUS : .enGB

                let wallStart = Date().timeIntervalSince1970
                let (audio, _) = try tts.generateAudio(
                    voice: voiceArray,
                    language: language,
                    text: text
                )
                let wallTime = Date().timeIntervalSince1970 - wallStart

                let sampleRate = Double(KokoroTTS.Constants.samplingRate)
                let audioDur = Double(audio.count) / sampleRate
                let rtf = audioDur > 0 ? (wallTime / audioDur) : 0

                let wavURL = try Self.saveWav(audio: audio, sampleRate: sampleRate)

                let result = RunResult(
                    text: text,
                    voice: voiceKey,
                    wallTimeSec: wallTime,
                    audioDurationSec: audioDur,
                    rtf: rtf,
                    wavURL: wavURL
                )

                DispatchQueue.main.async {
                    self.lastResult = result
                    self.lastWavURL = wavURL
                    self.status = String(
                        format: "RTF %.3f  (%.1f× realtime)  audio %.2f s  wall %.2f s",
                        rtf, rtf > 0 ? 1.0 / rtf : 0, audioDur, wallTime
                    )
                    self.isRunning = false
                }

                self.play(audio: audio, sampleRate: sampleRate)
            } catch {
                DispatchQueue.main.async {
                    self.status = "Synth error: \(error.localizedDescription)"
                    self.isRunning = false
                }
            }
        }
    }

    // MARK: - Playback

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
    }

    func playLastAgain() {
        guard let url = lastWavURL else { return }
        // Re-decode and play
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
        } catch {
            // ignore
        }
    }

    // MARK: - WAV save

    private static func saveWav(audio: [Float], sampleRate: Double) throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let ts = Int(Date().timeIntervalSince1970)
        let url = docs.appendingPathComponent("hike_\(ts).wav")
        // Use the package's built-in WAV writer rather than re-implementing.
        try AudioUtils.writeWavFile(samples: audio, sampleRate: sampleRate, fileURL: url)
        return url
    }
}

// MARK: - Result type

struct RunResult: Identifiable {
    let id = UUID()
    let text: String
    let voice: String
    let wallTimeSec: Double
    let audioDurationSec: Double
    /// wallTime / audioDuration; lower is better (< 1.0 = faster than realtime).
    let rtf: Double
    let wavURL: URL
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

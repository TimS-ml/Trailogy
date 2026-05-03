// ValidationRunner.swift
// Orchestrates one synthesis run: load fixture JSON from bundle, init the
// model provider with the chosen MLComputeUnits, call executeKokoroSynthesis,
// time it, save the WAV to Documents/, and surface a RunResult to the UI.

import CoreML
import Foundation
import KokoroPipeline

final class ValidationRunner: ObservableObject {

    // MARK: - Published UI state
    @Published private(set) var status: String = "Ready"
    @Published private(set) var results: [RunResult] = []
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var lastWavURL: URL?

    // MARK: - Internal cache (provider is reused if compute units don't change)
    private var provider: ConfigurableModelProvider?
    private var loadedComputeUnits: MLComputeUnits?

    private let workQueue = DispatchQueue(label: "com.lijuncheng16.HikeCompanion.validator",
                                          qos: .userInitiated)

    // MARK: - Public API

    func run(fixtureKey: String, computeUnits: MLComputeUnits) {
        guard !isRunning else { return }

        DispatchQueue.main.async {
            self.isRunning = true
            self.status = "Preparing…"
        }

        workQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.runSync(fixtureKey: fixtureKey, computeUnits: computeUnits)
            } catch {
                DispatchQueue.main.async {
                    self.status = "Error: \(error.localizedDescription)"
                    self.isRunning = false
                }
            }
        }
    }

    // MARK: - Heavy work (off main thread)

    private func runSync(fixtureKey: String, computeUnits: MLComputeUnits) throws {
        // 1) (Re-)init provider if compute units changed
        if loadedComputeUnits != computeUnits || provider == nil {
            DispatchQueue.main.async {
                self.status = "Loading models on \(Self.name(computeUnits))… (first run compiles, ~10–30 s)"
            }
            provider = ConfigurableModelProvider(
                modelsDir: ValidationRunner.modelsBundleDir(),
                computeUnits: computeUnits
            )
            loadedComputeUnits = computeUnits
        }
        guard let p = provider else {
            throw RunnerError.providerInitFailed
        }

        // 2) Load fixture + HNSF weights from bundle
        DispatchQueue.main.async { self.status = "Loading fixture \(fixtureKey)…" }
        let benchInput = try ValidationRunner.loadFixture(key: fixtureKey)
        let weights = try ValidationRunner.loadHnsfWeights()

        // 3) Run synthesis (this is the part we're timing)
        DispatchQueue.main.async { self.status = "Synthesising…" }
        let request = KokoroSynthesisRequest(
            inputIds: benchInput.input_ids,
            attentionMask: benchInput.attention_mask,
            refS: benchInput.ref_s,
            speed: benchInput.speed,
            seed: 42,
            warmModelsBeforeTiming: true,
            bucketDurationOverrideSeconds: benchInput.canonical_duration_s
        )

        var tensorDump: TensorDumpWriter? = nil
        let synth = try executeKokoroSynthesis(
            request: request,
            modelProvider: p,
            linearWeights: weights.linear_weights,
            linearBias: weights.linear_bias,
            tensorDump: &tensorDump
        )

        // 4) Save WAV
        let wavURL = try ValidationRunner.saveWav(
            audio: synth.audio,
            key: fixtureKey,
            computeUnits: computeUnits
        )

        // 5) Compute results, publish on main
        let canonicalDur = benchInput.canonical_duration_s ?? synth.audioDurationSeconds
        let rtf = canonicalDur > 0 ? (synth.wallTimeSeconds / canonicalDur) : -1
        let result = RunResult(
            computeUnits: Self.name(computeUnits),
            fixtureKey: fixtureKey,
            wallTimeSec: synth.wallTimeSeconds,
            audioDurationSec: canonicalDur,
            rtf: rtf,
            bucketSec: synth.bucketSeconds,
            timestamp: Date(),
            wavURL: wavURL
        )

        DispatchQueue.main.async {
            self.results.insert(result, at: 0)
            self.lastWavURL = wavURL
            if rtf > 0 {
                self.status = String(format: "RTF %.3f  (%.1f× realtime)", rtf, 1.0 / rtf)
            } else {
                self.status = "Done"
            }
            self.isRunning = false
        }
    }

    // MARK: - Helpers

    private static func modelsBundleDir() -> URL {
        // Folder reference "Models" is copied as-is into the .app bundle.
        Bundle.main.bundleURL.appendingPathComponent("Models")
    }

    private static func loadFixture(key: String) throws -> BenchInput {
        // Try direct lookup first; fall back to subdirectory.
        let candidates: [URL?] = [
            Bundle.main.url(forResource: key, withExtension: "json"),
            Bundle.main.url(forResource: key, withExtension: "json", subdirectory: "Fixtures"),
        ]
        guard let url = candidates.compactMap({ $0 }).first else {
            throw RunnerError.fixtureMissing(key: key)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(BenchInput.self, from: data)
    }

    private static func loadHnsfWeights() throws -> HnsfWeights {
        let candidates: [URL?] = [
            Bundle.main.url(forResource: "hnsf_weights", withExtension: "json"),
            Bundle.main.url(forResource: "hnsf_weights", withExtension: "json", subdirectory: "Fixtures"),
        ]
        guard let url = candidates.compactMap({ $0 }).first else {
            throw RunnerError.hnsfMissing
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(HnsfWeights.self, from: data)
    }

    private static func saveWav(audio: [Float], key: String, computeUnits: MLComputeUnits) throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let ts = Int(Date().timeIntervalSince1970)
        let filename = "\(key)_\(name(computeUnits))_\(ts).wav"
        let url = docs.appendingPathComponent(filename)
        try writeWavMono16(
            path: url.path,
            samples: audio,
            sampleRate: UInt32(PipelineConstants.sampleRate)
        )
        return url
    }

    static func name(_ cu: MLComputeUnits) -> String {
        switch cu {
        case .all: return "all"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        case .cpuOnly: return "cpuOnly"
        @unknown default: return "unknown"
        }
    }
}

// MARK: - Result type

struct RunResult: Identifiable {
    let id = UUID()
    let computeUnits: String
    let fixtureKey: String
    let wallTimeSec: Double
    let audioDurationSec: Double
    /// wallTime / audioDuration; lower is better. < 1.0 = faster than realtime.
    let rtf: Double
    let bucketSec: Int
    let timestamp: Date
    let wavURL: URL
}

// MARK: - Errors

enum RunnerError: LocalizedError {
    case providerInitFailed
    case fixtureMissing(key: String)
    case hnsfMissing

    var errorDescription: String? {
        switch self {
        case .providerInitFailed:
            return "Failed to initialize ConfigurableModelProvider."
        case .fixtureMissing(let key):
            return "Fixture \(key).json not in app bundle. Run scripts/prepare-fixtures.sh and rebuild."
        case .hnsfMissing:
            return "hnsf_weights.json not in app bundle. Run scripts/prepare-fixtures.sh and rebuild."
        }
    }
}

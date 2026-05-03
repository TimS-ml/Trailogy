// ContentView.swift
// Validator UI for mlalma/kokoro-ios. Text input, voice picker, speed
// slider, run button, results readout, live caption during playback,
// replay + share.

import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var runner = ValidationRunner()
    @State private var inputText: String = "The morning mist rose from the valley as we climbed the ridge."
    @State private var speed: Double = 1.0
    @State private var showShareSheet = false
    @State private var shareItems: [Any] = []

    var body: some View {
        NavigationStack {
            Form {
                Section("Status") {
                    Text(runner.status)
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Section("Input") {
                    TextField("Text to synthesize", text: $inputText, axis: .vertical)
                        .lineLimit(2...6)
                        .textFieldStyle(.roundedBorder)
                    Picker("Voice", selection: $runner.selectedVoice) {
                        ForEach(runner.voiceNames, id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .disabled(runner.voiceNames.isEmpty)
                    HStack {
                        Text("Speed")
                        Slider(value: $speed, in: 0.5...2.0, step: 0.05)
                        Text(String(format: "%.2f×", speed))
                            .font(.callout.monospaced())
                            .frame(width: 60, alignment: .trailing)
                    }
                }

                Section {
                    Button {
                        runner.synthesize(text: inputText, speed: Float(speed))
                    } label: {
                        HStack {
                            if runner.isRunning {
                                ProgressView().padding(.trailing, 6)
                            }
                            Text(runner.isRunning ? "Synthesising…" : "Synthesize")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!runner.isReady || runner.isRunning || inputText.isEmpty)
                }

                if !runner.currentCaption.isEmpty {
                    Section("Spoken so far") {
                        Text(runner.currentCaption)
                            .font(.body)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .fixedSize(horizontal: false, vertical: true)
                            .animation(.easeInOut(duration: 0.05), value: runner.currentCaption)
                    }
                }

                if let r = runner.lastResult {
                    Section("Last run") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(r.voice)  ·  \(String(format: "%.2f×", r.speed))")
                                .font(.subheadline.weight(.semibold))
                            Text(r.text)
                                .font(.callout)
                                .lineLimit(3)
                                .foregroundStyle(.secondary)
                            Text(String(format: "RTF %.3f   (%.1f× realtime)",
                                        r.rtf, r.rtf > 0 ? 1.0 / r.rtf : 0))
                                .font(.callout.monospaced())
                            Text(String(format: "wall %.2f s   audio %.2f s   chunks %d",
                                        r.wallTimeSec, r.audioDurationSec, r.chunkCount))
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        HStack {
                            Button("Play again") { runner.playLastAgain() }
                                .buttonStyle(.bordered)
                            Spacer()
                            Button("Share WAV") {
                                if let url = runner.lastWavURL {
                                    shareItems = [url]
                                    showShareSheet = true
                                }
                            }
                            .buttonStyle(.bordered)
                            .disabled(runner.lastWavURL == nil)
                        }
                    }
                }
            }
            .navigationTitle("HikeCompanion")
            .sheet(isPresented: $showShareSheet) {
                ShareSheet(items: shareItems)
            }
        }
    }
}

// MARK: - UIActivityViewController bridge

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

#Preview {
    ContentView()
}

// HikeCompanionApp.swift
// SwiftUI @main entry. Caps MLX GPU memory at launch — without this, MLX
// will grow its allocation as needed and we'll OOM once we add Gemma 4
// alongside Kokoro. Mirrors the limits used by mlalma's KokoroTestApp.

import MLX
import SwiftUI

@main
struct HikeCompanionApp: App {

    init() {
        // 50 MB cache (working scratch space for MLX kernels)
        GPU.set(cacheLimit: 50 * 1024 * 1024)
        // 900 MB hard ceiling on MLX GPU allocations.
        // Kokoro inference fits well under this; remaining iPhone Pro RAM
        // (~3–4 GB working set) is reserved for Gemma when we add it.
        GPU.set(memoryLimit: 900 * 1024 * 1024)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

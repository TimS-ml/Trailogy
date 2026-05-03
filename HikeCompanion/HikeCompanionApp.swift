// HikeCompanionApp.swift
// SwiftUI @main entry. Caps MLX GPU memory at launch so Kokoro + Gemma
// can coexist without unbounded growth → iOS jetsam.

import MLX
import SwiftUI

@main
struct HikeCompanionApp: App {

    init() {
        // No explicit Memory.memoryLimit cap with Gemma + Kokoro coexisting.
        // Reasoning: a hard ceiling forces MLX to allocate at the critical
        // path during inference, which can spike resident memory at exactly
        // the wrong moment (e.g. when Kokoro starts after Gemma finishes)
        // and trip iOS jetsam. Letting MLX size its own working set tends
        // to be steadier in practice — it grows the arena early and reuses
        // it. We can re-introduce a cap if we see runaway growth.
        //
        // Cache limit kept small — this is just kernel scratch space, not
        // tensor storage.
        Memory.cacheLimit = 100 * 1024 * 1024
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

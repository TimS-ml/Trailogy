// AppRouter.swift
// State container for the picker → detail → walking → journal flow.
// Mirrors the view-switching state machine in design/mockups.html.
//
// `screen` drives which top-level view is rendered. `currentTrail` is the
// trail the user picked; it's set when navigating from picker to detail.
// `debugVisible` is a sheet flag, separate from the main flow.

import SwiftUI

enum AppScreen: Equatable {
    case picker
    case detail
    case walking
    case journal
}

@MainActor
final class AppRouter: ObservableObject {
    @Published var screen: AppScreen = .picker
    @Published var currentTrail: Trail = TrailData.kildoo
    @Published var debugVisible: Bool = false

    /// In-memory set of trail IDs whose offline pack is "downloaded".
    /// Seeded at launch from each trail's `initiallyDownloaded` flag;
    /// gains entries when DetailView's CTA finishes its faux-download
    /// animation. Mirrors design/mockups.html's `t.downloaded` runtime
    /// flag — see design/README.md item 17.
    ///
    /// PoC scope: in-memory only, doesn't survive app relaunch. To
    /// ship real per-trail downloads, persist this set to UserDefaults
    /// (or build a proper DownloadService backed by URLSessionDownloadTask).
    @Published var downloadedTrailIDs: Set<String>

    init() {
        downloadedTrailIDs = Set(
            TrailData.all.filter(\.initiallyDownloaded).map(\.id)
        )
    }

    func isDownloaded(_ trail: Trail) -> Bool {
        downloadedTrailIDs.contains(trail.id)
    }

    func markDownloaded(_ trail: Trail) {
        downloadedTrailIDs.insert(trail.id)
    }

    func go(_ s: AppScreen) {
        withAnimation(.easeInOut(duration: 0.4)) {
            screen = s
        }
    }

    func choose(_ t: Trail) {
        currentTrail = t
        go(.detail)
    }

    func begin() { go(.walking) }
    func backToPicker() { go(.picker) }
    func endTour() { go(.journal) }
    func closeJournal() { go(.picker) }

    func openDebug()  { debugVisible = true  }
    func closeDebug() { debugVisible = false }
}

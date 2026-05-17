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

    /// When each trail was last walked. Populated by `endTour()` on
    /// tour completion (the moment WalkingView routes to JournalView).
    /// PickerView reads this to render the "Completed [date]" badge on
    /// a walked trail's card, and JournalView reads it for the recap
    /// meta line ("Completed May 16" instead of always "today").
    ///
    /// Mirrors `t.walked + t.walkedDate` from design/mockups.html
    /// (commit 8bf8889 — "Completed badge: stamp on tour finish, no
    /// hardcoded state"). Like `downloadedTrailIDs`, this is in-memory
    /// only and resets on app relaunch — the demo deliberately starts
    /// fresh with no completion history.
    @Published var walkedAt: [String: Date] = [:]

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

    /// `true` if the user has completed at least one tour of this trail
    /// in the current app session. Drives the picker's "Completed
    /// [date]" badge — see PickerView.statusBadge.
    func isWalked(_ trail: Trail) -> Bool {
        walkedAt[trail.id] != nil
    }

    /// Localized short date for the picker badge / recap meta. e.g.
    /// "May 16" — uses the user's locale via DateFormatter rather than
    /// a hardcoded format so it reads naturally in non-English locales
    /// too. Returns nil if the trail hasn't been walked.
    func walkedDateLabel(_ trail: Trail) -> String? {
        guard let date = walkedAt[trail.id] else { return nil }
        let f = DateFormatter()
        f.setLocalizedDateFormatFromTemplate("MMM d")
        return f.string(from: date)
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

    /// Called when the user reaches the tour's terminal `complete`
    /// phase (or taps End Tour from the More menu). Stamps the trail
    /// as walked with today's date, then routes to the Recap. The
    /// picker shows the resulting "Completed [date]" badge on next
    /// visit. Mirrors `markTrailWalked()` in design/mockups.html.
    func endTour() {
        walkedAt[currentTrail.id] = Date()
        go(.journal)
    }

    func closeJournal() { go(.picker) }

    func openDebug()  { debugVisible = true  }
    func closeDebug() { debugVisible = false }
}

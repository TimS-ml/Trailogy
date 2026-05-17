// JournalView.swift
// Post-tour Recap — what you learned, not what you did.
//
// Mockup: design/mockups.html → `.journal` view (current iteration).
// Per design/README.md item 16: the journal was reframed from a trip
// report (route map + per-stop photo cards + sightings list +
// share-when-connected button) into a knowledge digest.
//
// Visual treatment ported from upstream commit ad5a216 ("Recap
// redesign: trailmark, 9-category icons, dynamic per-trail content"):
//
//   ┌──────────────────────────────────────────────────────┐
//   │                                                  [X] │
//   │                       ▲                              │
//   │                     ▲   ▲          ← trailmark      │
//   │                                       (3 lime        │
//   │                                       rectangles =   │
//   │                                       US trail-end   │
//   │                                       blaze)         │
//   │                  Kildoo Trail                        │
//   │           May 16 · 2.0 mi · 1 hr · 5 stops           │
//   │                                                      │
//   │                  TAKEAWAYS                           │
//   │                                                      │
//   │ ┌──────────────────────────────────────────────  ⛰  ┐│
//   │ │ 320 million years                                  ││
//   │ │ Age of the sandstone in the layered cliffs ...     ││
//   │ └────────────────────────────────────────────────────┘│
//   │ ┌──────────────────────────────────────────────  ⚛  ┐│
//   │ │ Iron oxide ...                                     ││
//   │ └────────────────────────────────────────────────────┘│
//   │ (etc through 05, each with its own category icon)   │
//   └──────────────────────────────────────────────────────┘
//
// Content per trail lives on `Trail.learnings` (see TrailData.swift).
// Each Learning carries a `category` that selects the icon.

import SwiftUI

struct JournalView: View {
    @EnvironmentObject var router: AppRouter

    var trail: Trail { router.currentTrail }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            AppColor.screenBg.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .center, spacing: 0) {
                    recapHeader
                        .padding(.top, 92)
                        .padding(.horizontal, 22)
                        .padding(.bottom, 36)

                    takeawaysEyebrow
                        .padding(.bottom, 22)

                    discoveriesStream
                        .padding(.horizontal, 22)
                        .padding(.bottom, 48)
                }
                .frame(maxWidth: .infinity)
            }
            .scrollIndicators(.hidden)

            closeButton
                .padding(.top, 64)
                .padding(.trailing, 24)
        }
    }

    // MARK: - Recap header (centered stack: trailmark + name + meta)

    /// The recap's "stamp" composition. Three filled lime rectangles
    /// in a triangle = the actual trail-end blaze used on US hiking
    /// trails (Appalachian Trail convention). Brand-specific to
    /// Trailogy: the mark IS a trail mark. For people who know the
    /// convention it reads as "trail terminus"; for everyone else
    /// it's a deliberate lime triple-rectangle symbol.
    private var recapHeader: some View {
        VStack(spacing: 16) {
            trailmark
                .frame(width: 36, height: 40)
                .foregroundStyle(AppColor.lime)
                .shadow(color: AppColor.lime.opacity(0.25), radius: 8)

            Text(trail.name)
                .font(AppFont.sans(28, .bold))
                .foregroundStyle(AppColor.ink100)
                .tracking(-0.6)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)

            Text(metaStats)
                .font(AppFont.sans(12.5, .medium))
                .foregroundStyle(AppColor.ink60)
                .tracking(0.2)
        }
    }

    /// Three rounded lime rectangles in a triangle. Transcribed from
    /// design/mockups.html `.rm-mark` SVG (viewBox 36×40):
    ///   top:          x=14  y=1.5   w=8 h=14
    ///   bottom-left:  x=2.5 y=22.5  w=8 h=14
    ///   bottom-right: x=25.5 y=22.5 w=8 h=14
    /// All with corner radius 1.5.
    private var trailmark: some View {
        ZStack {
            // Top center
            RoundedRectangle(cornerRadius: 1.5)
                .frame(width: 8, height: 14)
                .position(x: 18, y: 8.5)
            // Bottom left
            RoundedRectangle(cornerRadius: 1.5)
                .frame(width: 8, height: 14)
                .position(x: 6.5, y: 29.5)
            // Bottom right
            RoundedRectangle(cornerRadius: 1.5)
                .frame(width: 8, height: 14)
                .position(x: 29.5, y: 29.5)
        }
    }

    /// "May 16 · 2.0 mi · 1 hr · 5 stops" — reads the walked date
    /// from the router (stamped on `endTour()`). Falls back to today
    /// if the user opened the journal without finishing a tour
    /// (e.g. via the picker's Journal link).
    private var metaStats: String {
        let dateStr = router.walkedDateLabel(trail) ?? todayLabel()
        let miles = trail.distanceMiles == floor(trail.distanceMiles)
            ? String(format: "%.0f", trail.distanceMiles)
            : String(format: "%.1f", trail.distanceMiles)
        return "\(dateStr) · \(miles) mi · \(formattedDuration) · \(trail.stops.count) stops"
    }

    private func todayLabel() -> String {
        let f = DateFormatter()
        f.setLocalizedDateFormatFromTemplate("MMM d")
        return f.string(from: Date())
    }

    /// "30 min" / "1 hr" / "1 hr 12 min" — same friendly format used
    /// elsewhere in the app (picker, tour completion).
    private var formattedDuration: String {
        let m = trail.durationMinutes
        if m < 60 { return "\(m) min" }
        let h = m / 60
        let r = m % 60
        return r == 0 ? "\(h) hr" : "\(h) hr \(r) min"
    }

    // MARK: - Takeaways section header

    /// Centered uppercase tracked "TAKEAWAYS" label, lime. Replaces
    /// the earlier giant "5 · Discoveries today" hero count — quieter
    /// section marker that doesn't compete with the trailmark above
    /// or the card headlines below.
    private var takeawaysEyebrow: some View {
        Text("Takeaways")
            .eyebrowStyle(AppColor.lime)
    }

    // MARK: - Discoveries stream (the learning cards)

    private var discoveriesStream: some View {
        VStack(spacing: 14) {
            ForEach(trail.learnings) { learning in
                learningCard(learning)
            }
        }
    }

    private func learningCard(_ learning: Learning) -> some View {
        ZStack(alignment: .topTrailing) {
            VStack(alignment: .leading, spacing: 10) {
                Text(learning.anchor)
                    .font(AppFont.sans(22, .bold))
                    .foregroundStyle(AppColor.ink100)
                    .tracking(-0.4)
                    .lineSpacing(2)
                    .padding(.trailing, 50)  // clearance for the 42px corner icon
                    .fixedSize(horizontal: false, vertical: true)

                Text(learning.body)
                    .font(AppFont.sans(14, .medium))
                    .foregroundStyle(AppColor.ink100.opacity(0.78))
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 22)
            .padding(.top, 24)
            .padding(.bottom, 22)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                LinearGradient(
                    colors: [
                        AppColor.ink100.opacity(0.045),
                        AppColor.ink100.opacity(0.015),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 16, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(AppColor.ink100.opacity(0.12), lineWidth: 1)
            )

            // Category icon, top-right. 42px, lime at 85 % with a soft
            // halo so it reads against the lime-tinted card surface.
            categoryIcon(learning.category)
                .font(.system(size: 26, weight: .regular))
                .foregroundStyle(AppColor.lime.opacity(0.85))
                .frame(width: 42, height: 42)
                .shadow(color: AppColor.lime.opacity(0.18), radius: 6)
                .padding(.top, 14)
                .padding(.trailing, 14)
        }
    }

    /// SF Symbol per learning category. Picked for "reads instantly"
    /// rather than a 1:1 visual translation of the upstream hand-drawn
    /// SVGs (`design/mockups.html` CATEGORY_ICONS) — SF Symbols give
    /// us consistent iOS look, Dynamic Type, and tint handling.
    @ViewBuilder
    private func categoryIcon(_ cat: LearningCategory) -> some View {
        switch cat {
        case .geology:      Image(systemName: "square.stack.3d.up")
        case .water:        Image(systemName: "drop")
        case .plant:        Image(systemName: "leaf")
        case .wildlife:     Image(systemName: "bird")
        case .history:      Image(systemName: "doc.text")
        case .architecture: Image(systemName: "building.columns")
        case .sky:          Image(systemName: "sun.max")
        case .chemistry:    Image(systemName: "atom")
        case .other:        Image(systemName: "asterisk")
        }
    }

    // MARK: - Close button

    private var closeButton: some View {
        Button {
            router.closeJournal()
        } label: {
            Image(systemName: "xmark")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(AppColor.ink100)
                .frame(width: 32, height: 32)
                .background(AppColor.glassDark88, in: Circle())
                .overlay(Circle().stroke(AppColor.hairlineHi, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    JournalView()
        .environmentObject({
            let r = AppRouter()
            r.currentTrail = TrailData.kildoo
            r.walkedAt[TrailData.kildoo.id] = Date()
            return r
        }())
}

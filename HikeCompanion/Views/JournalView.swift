// JournalView.swift
// Post-tour Recap — what you learned, not what you did.
//
// Mockup: design/mockups.html → `.journal` view (current iteration).
// Per design/README.md item 16: the journal was reframed from a trip
// report (route map + per-stop photo cards + sightings list +
// share-when-connected button) into a knowledge digest. Each card is
// "a museum catalog entry" anchored by a hero number/date/quantity.
//
// Layout:
//   ┌──────────────────────────────────────────────────────┐
//   │ ⊙ Kildoo Trail                                  [X] │
//   │   May 3 · 2.0 mi · 1 hr · 5 stops                    │
//   ├──────────────────────────────────────────────────────┤
//   │                                                      │
//   │                       5                              │
//   │                Discoveries today                     │
//   │                                                      │
//   ├──────────────────────────────────────────────────────┤
//   │ ┌──────────────────────────────────────────────  01 ┐│
//   │ │ 320 million years                                  ││
//   │ │ Age of the sandstone in the layered cliffs. The    ││
//   │ │ orange streaks are iron oxide leached out of the   ││
//   │ │ rock by groundwater over geologic time.            ││
//   │ └────────────────────────────────────────────────────┘│
//   │ ┌──────────────────────────────────────────────  02 ┐│
//   │ │ Iron oxide ...                                     ││
//   │ └────────────────────────────────────────────────────┘│
//   │ (etc through 05)                                     │
//   └──────────────────────────────────────────────────────┘
//
// Content per trail lives on `Trail.learnings` (see TrailData.swift).
// Currently 5 cards per trail, curator-authored.

import SwiftUI

struct JournalView: View {
    @EnvironmentObject var router: AppRouter

    var trail: Trail { router.currentTrail }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            AppColor.screenBg.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    recapMeta
                        .padding(.top, 64)
                        .padding(.horizontal, 22)
                        .padding(.bottom, 22)

                    // Hairline under the meta header
                    Rectangle()
                        .frame(height: 1)
                        .foregroundStyle(AppColor.ink100.opacity(0.10))
                        .padding(.horizontal, 22)

                    discoveryHero
                        .padding(.top, 38)
                        .padding(.bottom, 26)

                    discoveriesStream
                        .padding(.horizontal, 22)
                        .padding(.bottom, 48)
                }
            }
            .scrollIndicators(.hidden)

            closeButton
                .padding(.top, 64)
                .padding(.trailing, 24)
        }
    }

    // MARK: - Recap meta (compact horizontal header)

    private var recapMeta: some View {
        HStack(alignment: .center, spacing: 14) {
            // Lime check seal — like a wax stamp confirming the loop closed.
            ZStack {
                Circle()
                    .fill(AppColor.lime.opacity(0.10))
                    .overlay(Circle().stroke(AppColor.lime.opacity(0.55), lineWidth: 1))
                    .frame(width: 38, height: 38)
                Image(systemName: "checkmark")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(AppColor.lime)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(trail.name)
                    .font(AppFont.sans(17, .bold))
                    .foregroundStyle(AppColor.ink100)
                    .tracking(-0.3)
                Text(metaStats)
                    .font(AppFont.sans(12, .medium))
                    .foregroundStyle(AppColor.ink60)
            }

            Spacer(minLength: 0)
        }
    }

    private var metaStats: String {
        let f = DateFormatter()
        f.dateFormat = "MMM d"
        let dateStr = f.string(from: Date())
        let miles = trail.distanceMiles == floor(trail.distanceMiles)
            ? String(format: "%.0f", trail.distanceMiles)
            : String(format: "%.1f", trail.distanceMiles)
        return "\(dateStr) · \(miles) mi · \(formattedDuration) · \(trail.stops.count) stops"
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

    // MARK: - Discovery hero (big lime count)

    private var discoveryHero: some View {
        VStack(spacing: 10) {
            Text("\(trail.learnings.count)")
                .font(AppFont.sans(80, .bold))
                .tracking(-3.0)
                .foregroundStyle(AppColor.lime)
                .shadow(color: AppColor.lime.opacity(0.20), radius: 26)
                .monospacedDigit()
            Text("Discoveries today")
                .eyebrowStyle(AppColor.ink60)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Discoveries stream (the learning cards)

    private var discoveriesStream: some View {
        VStack(spacing: 14) {
            ForEach(Array(trail.learnings.enumerated()), id: \.element.id) { idx, learning in
                learningCard(learning: learning, index: idx + 1)
            }
        }
    }

    private func learningCard(learning: Learning, index: Int) -> some View {
        ZStack(alignment: .topTrailing) {
            VStack(alignment: .leading, spacing: 12) {
                Text(learning.anchor)
                    .font(AppFont.sans(28, .bold))
                    .foregroundStyle(AppColor.ink100)
                    .tracking(-0.7)
                    .lineSpacing(2)
                    .padding(.trailing, 36)  // breathing room for corner number
                    .fixedSize(horizontal: false, vertical: true)

                Text(learning.body)
                    .font(AppFont.sans(14.5, .medium))
                    .foregroundStyle(AppColor.ink100.opacity(0.95))
                    .lineSpacing(4)
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

            // "01"–"05" corner number, lime, dim.
            Text(String(format: "%02d", index))
                .font(AppFont.sans(10.5, .heavy))
                .tracking(1.8)
                .foregroundStyle(AppColor.lime.opacity(0.55))
                .monospacedDigit()
                .padding(.top, 14)
                .padding(.trailing, 18)
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
            return r
        }())
}

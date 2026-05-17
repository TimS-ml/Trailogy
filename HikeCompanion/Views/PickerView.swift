// PickerView.swift
// "Trails near you" — the entry screen.
// Mockup: design/mockups.html → .picker view.

import SwiftUI

struct PickerView: View {
    @EnvironmentObject var router: AppRouter

    var body: some View {
        ZStack(alignment: .top) {
            AppColor.screenBg.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // Header — mirrors design/mockups.html `.picker-head`.
                    // Replaces the earlier "Nature, the best teacher /
                    // Trails near you / Pittsburgh" stack with a single
                    // softer question paired with a lime location pin —
                    // the visual answer to "where". The discipline of
                    // absences (no profile chip, no eyebrow, no city
                    // label) is part of the design.
                    HStack(spacing: 10) {
                        Image(systemName: "location.fill")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(AppColor.lime)
                        Text("Where should Trailogy take you?")
                            .font(AppFont.sans(18, .regular))
                            .foregroundStyle(AppColor.ink100)
                            .tracking(-0.2)
                    }
                    .padding(.horizontal, 22)
                    .padding(.top, 6)
                    .padding(.bottom, 26)

                    // Cards
                    VStack(spacing: 18) {
                        ForEach(TrailData.all) { trail in
                            TrailCard(trail: trail) {
                                router.choose(trail)
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 24)

                    // Footer — visible Journal + Debug entries side-by-side.
                    // Subtle but discoverable, matching the picker's
                    // ink60 secondary-text style.
                    HStack(spacing: 32) {
                        Spacer()
                        Button {
                            router.go(.journal)
                        } label: {
                            Text("Journal")
                                .font(AppFont.sans(13, .medium))
                                .foregroundStyle(AppColor.ink60)
                        }
                        .buttonStyle(.plain)

                        Rectangle()
                            .frame(width: 1, height: 11)
                            .foregroundStyle(AppColor.ink25)

                        Button {
                            router.openDebug()
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "ladybug.fill")
                                    .font(.system(size: 11, weight: .semibold))
                                Text("Debug")
                                    .font(AppFont.sans(13, .medium))
                            }
                            .foregroundStyle(AppColor.ink60)
                        }
                        .buttonStyle(.plain)
                        Spacer()
                    }
                    .padding(.vertical, 14)

                    Color.clear.frame(height: 32)
                }
                // No extra top padding — the iOS safe-area inset
                // already provides dynamic-island clearance (~59 pt on
                // iPhone 15 Pro). Anything on top of that is dead
                // space. Header sits as close to the dynamic island as
                // possible without clipping.
                .padding(.top, 0)
            }
            .scrollIndicators(.hidden)
        }
    }
}

// MARK: - TrailCard

private struct TrailCard: View {
    @EnvironmentObject var router: AppRouter
    let trail: Trail
    let onTap: () -> Void

    /// Status is now derived from router runtime state (`walkedAt`)
    /// rather than from a hardcoded table — see design/README.md
    /// commit 8bf8889. Mirrors the mockup's `renderPickerBadges()`,
    /// which injects the "Completed [date]" badge on every walked
    /// trail at picker render time, with no demo-baked completions.
    var status: TrailStatus { TrailData.status(for: trail, router: router) }

    var body: some View {
        Button(action: onTap) {
            // Overlay layout — photo fills the card, text sits in a
            // near-opaque dark band at the bottom that overlaps the
            // photo. The "band" is a steep gradient that goes from
            // transparent at the photo's top through ~95% opaque
            // black where the text lives. Photo is visible in the
            // upper ~50 % of the card; the bottom ~50 % is heavily
            // darkened so the type stays legible regardless of which
            // colors dominate that part of any given photo.
            //
            // History: a first overlay attempt with a softer gradient
            // (d23ff89 → cf46f3e) made text bleed on brighter photos.
            // This iteration takes the dark band all the way to ~95 %
            // and adds a 0.6-opacity 4-pt text shadow as a fallback.
            ZStack(alignment: .bottomLeading) {
                photoLayer
                scrimLayer
                textOverlay
                    .padding(.horizontal, 18)
                    .padding(.bottom, 16)
                if case .walked(let date) = status {
                    walkedBadge(date: date)
                        .padding(14)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 275)
            .background(AppColor.ink25)
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(AppColor.ink100.opacity(0.07), lineWidth: 1)
            )
        }
        .buttonStyle(CardPressStyle())
    }

    /// Bottom layer — the trail's cover photo, filling the entire card.
    /// Cropped centre-out by `.scaledToFill().clipped()` regardless of
    /// source aspect.
    private var photoLayer: some View {
        AsyncImage(url: trail.coverImageURL) { phase in
            switch phase {
            case .empty:
                AppColor.ink25
            case .success(let img):
                img.resizable().scaledToFill()
            case .failure:
                AppColor.ink25
            @unknown default:
                AppColor.ink25
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipped()
    }

    /// Middle layer — top-to-bottom gradient. Aggressive on purpose:
    /// stays fully transparent through the top half so the photo
    /// reads, then ramps to near-opaque black across the bottom 40 %
    /// where the text overlay lives. The text effectively sits on a
    /// solid dark band that the photo only barely shows through.
    private var scrimLayer: some View {
        LinearGradient(
            stops: [
                .init(color: .black.opacity(0.0),  location: 0.30),
                .init(color: .black.opacity(0.45), location: 0.55),
                .init(color: .black.opacity(0.85), location: 0.80),
                .init(color: .black.opacity(0.95), location: 1.0),
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        .allowsHitTesting(false)
    }

    /// Top layer — text block anchored bottom-left. Region eyebrow
    /// (lime) / trail name / one-line tagline / stats row. Drop
    /// shadow on each line as a safety net for the rare bright spot
    /// that still bleeds through the scrim.
    private var textOverlay: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(trail.region.uppercased())
                .font(AppFont.sans(10, .semibold))
                .tracking(1.6)
                .foregroundStyle(AppColor.lime)
                .lineLimit(1)
                .minimumScaleFactor(0.85)

            Text(trail.name)
                .font(AppFont.sans(22, .bold))
                .foregroundStyle(AppColor.ink100)
                .tracking(-0.4)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .padding(.top, 1)

            Text(trail.summary)
                .font(AppFont.sans(13, .regular))
                .foregroundStyle(AppColor.ink100.opacity(0.92))
                .lineSpacing(1)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 2)

            HStack(spacing: 8) {
                Text("\(formattedMiles) mi")
                circleDot
                Text(trail.difficulty)
                circleDot
                Text(durationLabel)
            }
            .font(AppFont.sans(12.5, .medium))
            .foregroundStyle(AppColor.ink100.opacity(0.95))
            .padding(.top, 4)
            .lineLimit(1)
            .minimumScaleFactor(0.85)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .shadow(color: .black.opacity(0.55), radius: 4, x: 0, y: 1)
    }

    /// "60 min" or "1 hr" or "1 hr 15 min" — friendlier than always
    /// rendering minutes when the trail is over an hour.
    private var durationLabel: String {
        let m = trail.durationMinutes
        if m >= 60 {
            let h = m / 60
            let r = m % 60
            return r == 0 ? "\(h) hr" : "\(h) hr \(r) min"
        }
        return "\(m) min"
    }

    private var formattedMiles: String {
        let v = trail.distanceMiles
        return v == floor(v) ? String(format: "%.0f", v) : String(format: "%.1f", v)
    }

    private var circleDot: some View {
        Circle()
            .frame(width: 3, height: 3)
            .foregroundStyle(AppColor.ink100.opacity(0.55))
    }

    /// Top-right "Completed [date]" pill, shown only on a trail walked
    /// in this session. Picker cards are otherwise pure choice — no
    /// "Ready" or "Download · MB" pill (those retired in design/README.md
    /// items 17 & 23).
    private func walkedBadge(date: String) -> some View {
        Text("Completed \(date)")
            .font(AppFont.sans(10.5, .semibold))
            .tracking(0.6)
            .textCase(.uppercase)
            .foregroundStyle(AppColor.ink100.opacity(0.85))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.55), in: Capsule())
    }
}

private struct CardPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.985 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview {
    PickerView()
        .environmentObject(AppRouter())
}

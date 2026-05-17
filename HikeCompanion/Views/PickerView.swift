// PickerView.swift
// "Trails near you" — the entry screen.
// Mockup: design/mockups.html → .picker view.

import SwiftUI

struct PickerView: View {
    @EnvironmentObject var router: AppRouter

    /// Lime ambient glow — two stacked radial gradients originating
    /// from the top-left, behind the header. Mirrors the mockup's
    /// `.picker { background: radial-gradient(...), radial-gradient(...) }`.
    /// Inner spot is more intense and focused around the lime icon;
    /// outer halo trails off across the upper-third of the screen
    /// for a broader, calmer warmth.
    private var limeGlow: some View {
        ZStack {
            RadialGradient(
                colors: [AppColor.lime.opacity(0.10), .clear],
                center: UnitPoint(x: 0.18, y: 0.02),
                startRadius: 0,
                endRadius: 160
            )
            RadialGradient(
                colors: [AppColor.lime.opacity(0.04), .clear],
                center: UnitPoint(x: 0.18, y: 0.02),
                startRadius: 0,
                endRadius: 400
            )
        }
    }

    var body: some View {
        ZStack(alignment: .top) {
            AppColor.screenBg.ignoresSafeArea()

            // Lime ambient glow — soft inner spot + wider outer halo
            // originating from the top-left of the picker, sitting
            // behind the heading. Mirrors design/mockups.html `.picker`
            // background. Subtle but broad: the inner spot warms the
            // header area, the halo trails off across the upper third
            // of the screen without ever feeling saturated.
            limeGlow
                .ignoresSafeArea()
                .allowsHitTesting(false)

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
                            .foregroundStyle(AppColor.lime.opacity(0.9))
                        Text("Where should Trailogy take you?")
                            // 19 pt instead of the mockup's literal
                            // 17 px because iOS substitutes SF Pro
                            // for the mockup's Inter (Theme.swift
                            // explains) — SF Pro has a smaller
                            // x-height, so the same numeric point
                            // size reads smaller. Bumping by ~12 %
                            // lands the visual size in the same
                            // ballpark as the mockup's rendering.
                            .font(AppFont.sans(19, .regular))
                            .foregroundStyle(AppColor.ink100)
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

                    // Footer — single Debug entry. The earlier "Journal"
                    // shortcut was removed: with the new dynamic-walked-
                    // badge flow (`router.walkedAt`), each picker card
                    // surfaces its own "Completed [date]" badge that
                    // taps through to that trail's Recap. A standalone
                    // Journal link from the picker would just open the
                    // currently-selected trail's recap with no walked
                    // history, which isn't useful.
                    HStack {
                        Spacer()
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
                // 12-pt margin past the iOS safe-area inset so the
                // header sits a touch below the dynamic island instead
                // of right against it — matches the mockup's
                // `.picker-head { padding: 6px ... }` plus the 70-px
                // top-of-`.picker`, which together put the question
                // ~76 px from the top of the browser viewport. iOS
                // safe area + 12 pt lands in the same visual ballpark.
                .padding(.top, 12)
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
            // 210 pt — close to the mockup's `aspect-ratio: 16/9` at
            // typical iPhone widths (361-pt usable card width gives
            // ~203 pt at 16:9). Slight upsize past 203 because the
            // iOS text overlay block is ~10 pt taller than the
            // mockup's (SF Pro substitution carries a touch more
            // vertical space), so the visual proportion still reads
            // like the mockup's hero photo with a tidy text band.
            .frame(height: 210)
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
    ///
    /// Type sizes are all ~10–15 % above the mockup's literal px
    /// values to compensate for iOS's SF Pro vs the mockup's Inter
    /// (see Theme.swift). The mockup specs are: region 10, name 24,
    /// tagline 13, stats 13; iOS uses 11.5 / 26 / 14.5 / 14 to
    /// reach the same visual weight.
    private var textOverlay: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(trail.region.uppercased())
                .font(AppFont.sans(11.5, .semibold))
                .tracking(1.8)
                .foregroundStyle(AppColor.lime)
                .lineLimit(1)
                .minimumScaleFactor(0.85)

            Text(trail.name)
                .font(AppFont.sans(26, .bold))
                .foregroundStyle(AppColor.ink100)
                .tracking(-0.52)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .padding(.top, 2)

            Text(trail.summary)
                .font(AppFont.sans(14.5, .regular))
                .foregroundStyle(AppColor.ink100.opacity(0.92))
                .lineSpacing(1)
                // 1-line tagline (was 2) — guarantees the stats row
                // below has predictable space inside the card height
                // without minimumScaleFactor collapse antics.
                .lineLimit(1)
                .truncationMode(.tail)
                .padding(.top, 4)

            HStack(spacing: 10) {
                Text("\(formattedMiles) mi")
                circleDot
                Text(trail.difficulty)
                circleDot
                Text(durationLabel)
            }
            .font(AppFont.sans(14, .medium))
            .foregroundStyle(AppColor.ink100.opacity(0.95))
            .padding(.top, 6)
            .lineLimit(1)
            // Drop minimumScaleFactor — with lineLimit(1) it can
            // interact with SwiftUI's layout pass and collapse the
            // HStack to zero height when the width is constrained.
            // The stats text always fits at 14 pt on any iPhone size
            // we care about, so scaling isn't actually needed.
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

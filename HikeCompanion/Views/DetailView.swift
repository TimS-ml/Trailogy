// DetailView.swift
// Pre-tour detail screen: header (back arrow + trail name), full-screen
// trail map, bottom action card with stats + Begin button.
// Mockup: design/mockups.html → .detail view (.dm-* classes).

import SwiftUI

struct DetailView: View {
    @EnvironmentObject var router: AppRouter

    var trail: Trail { router.currentTrail }

    /// Demo-mode framing alert. Tapping Begin presents a system iOS
    /// alert explaining the location-based vs auto-advance behavior
    /// (the production app would gate stop unlocks on Core Location;
    /// this build can't be at the trail, so it ticks through on a
    /// timer). See design/README.md item 15.
    @State private var showBeginAlert: Bool = false

    /// The three CTA states from the mockup (design/README.md item 17):
    /// `download` shows "Download · 68 MB" + arrow icon; `downloading`
    /// shows a dark progress fill + percentage and no-ops on tap;
    /// `ready` is the familiar "Begin" + play icon that opens the
    /// demo-mode alert. Per-trail state is sourced from the router's
    /// `downloadedTrailIDs` set on appear and after a faux-download
    /// completes.
    private enum CTAState: Equatable { case download, downloading, ready }
    @State private var ctaState: CTAState = .ready
    @State private var ctaPercent: Double = 0
    @State private var downloadTask: Task<Void, Never>? = nil

    var body: some View {
        ZStack {
            AppColor.mapBg.ignoresSafeArea()

            VStack(spacing: 0) {
                topHeader
                mapCanvas
                bottomAction
            }
        }
        // Demo-mode framing — fires on every Begin tap.
        // Native SwiftUI .alert renders as the system iOS alert, so
        // the chrome (rounded glass card, button row, blur backdrop)
        // matches design/mockups.html `.ios-alert` for free. Tint is
        // applied to the alert presenter so the primary "Begin Tour"
        // button picks up the lime accent.
        //
        // Production path: replace the time-based phaseTimer in
        // WalkingView with Core Location region monitoring; this
        // alert becomes a real error path for GPS-denied / off-trail
        // cases, not the default Begin flow. See design/README.md
        // item 15.
        .alert("Tours are location-based", isPresented: $showBeginAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Begin Tour") {
                router.begin()
            }
        } message: {
            Text("On the trail, stops play when you arrive.\nThis demo will auto-advance.")
        }
        .tint(AppColor.lime)
        .onAppear { syncCTAState() }
        .onChange(of: trail.id) { _, _ in syncCTAState() }
        .onDisappear {
            downloadTask?.cancel()
            downloadTask = nil
        }
    }

    // MARK: - Top header (mockup: .dm-top)

    /// Three-line magazine stack: location eyebrow → trail name (hero) →
    /// summary subtitle (centered). The summary moved here from the
    /// bottom action card per the upstream design — qualitative info
    /// ("where + what + what kind of") all sits at the top together;
    /// the bottom card stays focused on numbers + the commit.
    private var topHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            Button {
                router.backToPicker()
            } label: {
                Image(systemName: "chevron.left")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(AppColor.ink100)
                    .frame(width: 32, height: 32)
                    .background(.black.opacity(0.4), in: Circle())
            }

            VStack(spacing: 0) {
                Text(trail.parkLocation.uppercased())
                    .font(AppFont.sans(10, .heavy))
                    .tracking(1.8)
                    .foregroundStyle(AppColor.ink60)
                    .padding(.bottom, 6)

                Text(trail.name)
                    .font(AppFont.sans(22, .bold))
                    .foregroundStyle(AppColor.ink100)
                    .tracking(-0.44)
                    .lineSpacing(1)
                    .multilineTextAlignment(.center)

                Text(trail.summary)
                    .font(AppFont.sans(13, .regular))
                    .foregroundStyle(AppColor.ink100.opacity(0.78))
                    .lineSpacing(2)
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 8)
                    .padding(.horizontal, 4)
            }
            .frame(maxWidth: .infinity)

            // Spacer to balance the back button so the title centers
            // correctly. Mirrors the mockup's empty 32-px div.
            Color.clear.frame(width: 32, height: 32)
        }
        .padding(.horizontal, 18)
        .padding(.top, 60)
        .padding(.bottom, 14)
        .overlay(alignment: .bottom) {
            // Hairline divider into the map — mirrors the .dm-top
            // border-bottom and the matching one on .dm-action so the
            // layout reads as three zones (title / map / action).
            Rectangle()
                .frame(height: 1)
                .foregroundStyle(AppColor.ink100.opacity(0.06))
        }
    }

    // MARK: - Map canvas (mockup: .dm-canvas)

    private var mapCanvas: some View {
        ZStack {
            // Subtle radial highlights from the mockup.
            RadialGradient(
                colors: [AppColor.lime.opacity(0.05), .clear],
                center: UnitPoint(x: 0.5, y: 0.3),
                startRadius: 20,
                endRadius: 200
            )

            // Edge-to-edge — no inset padding around the map. The
            // top header's hairline divider and the bottom action
            // card's hairline divider are the layout's "three-zone"
            // dividers; padding around the map would re-introduce a
            // visible margin between those dividers and the tiles.
            TrailMapView(trail: trail, activeStop: 1, passedThroughStop: 0)
        }
    }

    // MARK: - Bottom action card (mockup: .dm-action)

    /// "Numbers + the commit." Summary deliberately not here — it's in
    /// the top header now. Stats row uses bold values inline with
    /// ink-80 suffix text, larger than the previous iteration (16 pt
    /// vs 13.5 pt) so the row has real weight under the map.
    private var bottomAction: some View {
        VStack(spacing: 0) {
            statsRow
                .padding(.bottom, 18)
            ctaButton
        }
        .padding(.horizontal, 22)
        .padding(.top, 18)
        .padding(.bottom, 22)
        .background(
            Color(red: 15/255, green: 16/255, blue: 13/255).opacity(0.95)
                .overlay(
                    Rectangle()
                        .frame(height: 1)
                        .foregroundStyle(AppColor.hairline),
                    alignment: .top
                )
        )
    }

    /// "2.0 mi · 1 hr · 5 stops · Moderate" — bold value + light unit.
    /// Mirrors `.dm-stats b` (ink-100 bold) interleaved with non-bold
    /// suffix (ink-80), divided by 3.5-pt ink-40 dots.
    private var statsRow: some View {
        HStack(spacing: 12) {
            statPart(value: formattedMiles, suffix: "mi")
            dmDot
            statPart(value: durationLabel.0, suffix: durationLabel.1)
            dmDot
            statPart(value: "\(trail.stopCount)", suffix: "stops")
            dmDot
            Text(trail.difficulty)
                .font(AppFont.sans(16, .medium))
                .foregroundStyle(AppColor.ink80)
                .tracking(-0.16)
        }
        .frame(maxWidth: .infinity)
    }

    /// Bold value (ink-100) directly followed by a non-bold suffix
    /// (ink-80) — both at the same point size so they read as one
    /// phrase like "2.0 mi" rather than two columns.
    private func statPart(value: String, suffix: String) -> some View {
        HStack(spacing: 4) {
            Text(value)
                .font(AppFont.sans(16, .bold))
                .foregroundStyle(AppColor.ink100)
            Text(suffix)
                .font(AppFont.sans(16, .medium))
                .foregroundStyle(AppColor.ink80)
        }
        .tracking(-0.16)
    }

    /// 3.5-pt round dot between stat groups. Mirrors `.dm-stats .d-dot`.
    private var dmDot: some View {
        Circle()
            .frame(width: 3.5, height: 3.5)
            .foregroundStyle(AppColor.ink40)
    }

    // MARK: - State-aware CTA (mockup: design/mockups.html .cta-btn)

    /// The lime CTA capsule. Three visual states share one button so the
    /// position never jumps: dark progress fill grows L→R during download,
    /// then snaps to "Begin" + play icon. See design/README.md item 17.
    private var ctaButton: some View {
        Button {
            tapCTA()
        } label: {
            HStack(spacing: 10) {
                ctaIcon
                Text(ctaLabel)
                    .font(AppFont.sans(17, .semibold))
                    .foregroundStyle(AppColor.limeText)
                    .monospacedDigit()
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 18)
            .background(
                ZStack(alignment: .leading) {
                    AppColor.lime
                    GeometryReader { geo in
                        Rectangle()
                            .fill(.black.opacity(0.20))
                            .frame(width: geo.size.width * CGFloat(ctaPercent / 100.0))
                    }
                    .allowsHitTesting(false)
                }
            )
            .clipShape(Capsule())
        }
        .buttonStyle(LimePressStyle())
        .disabled(ctaState == .downloading)
        .animation(.easeInOut(duration: 0.2), value: ctaState)
    }

    @ViewBuilder
    private var ctaIcon: some View {
        switch ctaState {
        case .download:
            Image(systemName: "arrow.down.to.line")
                .font(.system(size: 15, weight: .heavy))
                .foregroundStyle(AppColor.limeText)
        case .ready:
            Image(systemName: "play.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(AppColor.limeText)
        case .downloading:
            EmptyView()
        }
    }

    private var ctaLabel: String {
        switch ctaState {
        case .download:    return "Download · \(trail.downloadSize)"
        case .downloading: return "\(Int(ctaPercent.rounded()))%"
        case .ready:       return "Begin"
        }
    }

    private func tapCTA() {
        switch ctaState {
        case .download:    startDownload()
        case .downloading: break // no-op while in flight
        case .ready:       showBeginAlert = true
        }
    }

    /// Mockup CTA: 110ms ticks, +6..+13% per tick, 260ms settle at 100%
    /// before flipping to .ready. The faux progress is decorative — the
    /// models are bundled at install — but the affordance teaches the
    /// "offline pack" mental model the production app would use.
    private func startDownload() {
        downloadTask?.cancel()
        ctaPercent = 0
        withAnimation(.easeInOut(duration: 0.2)) {
            ctaState = .downloading
        }
        downloadTask = Task { @MainActor in
            while ctaPercent < 100 && !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(110))
                if Task.isCancelled { return }
                let next = min(100, ctaPercent + 6 + Double.random(in: 0...7))
                withAnimation(.linear(duration: 0.12)) {
                    ctaPercent = next
                }
            }
            try? await Task.sleep(for: .milliseconds(260))
            if Task.isCancelled { return }
            router.markDownloaded(trail)
            withAnimation(.easeInOut(duration: 0.25)) {
                ctaState = .ready
                ctaPercent = 0
            }
        }
    }

    /// Reset CTA to match the trail's current download status. Called on
    /// appear and whenever the user picks a different trail (router
    /// re-uses the same DetailView instance for all three trails).
    private func syncCTAState() {
        downloadTask?.cancel()
        downloadTask = nil
        ctaPercent = 0
        ctaState = router.isDownloaded(trail) ? .ready : .download
    }

    private var formattedMiles: String {
        let v = trail.distanceMiles
        return v == floor(v) ? String(format: "%.0f", v) : String(format: "%.1f", v)
    }

    /// Minutes → ("1", "hr") or ("45", "min")
    private var durationLabel: (String, String) {
        let m = trail.durationMinutes
        if m >= 60 && m % 60 == 0 {
            return ("\(m/60)", m == 60 ? "hr" : "hrs")
        } else if m >= 60 {
            let h = Double(m) / 60.0
            return (String(format: "%.1f", h), "hr")
        } else {
            return ("\(m)", "min")
        }
    }

}

struct LimePressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .opacity(configuration.isPressed ? 0.9 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview {
    DetailView()
        .environmentObject({
            let r = AppRouter()
            r.currentTrail = TrailData.kildoo
            return r
        }())
}

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
                // Top nav: back arrow + centered title
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

                    VStack(spacing: 2) {
                        Text(trail.name)
                            .font(AppFont.sans(17, .bold))
                            .foregroundStyle(AppColor.ink100)
                            .tracking(-0.3)
                        Text(trail.parkLocation.uppercased())
                            .font(AppFont.sans(10, .heavy))
                            .tracking(1.8)
                            .foregroundStyle(AppColor.ink60)
                    }
                    .frame(maxWidth: .infinity)

                    // Spacer to balance the back button
                    Color.clear.frame(width: 32, height: 32)
                }
                .padding(.horizontal, 18)
                .padding(.top, 60)
                .padding(.bottom, 8)

                // Full-screen trail map
                ZStack {
                    // Subtle radial highlights from the mockup
                    RadialGradient(
                        colors: [AppColor.lime.opacity(0.05), .clear],
                        center: UnitPoint(x: 0.5, y: 0.3),
                        startRadius: 20,
                        endRadius: 200
                    )

                    TrailMapView(trail: trail, activeStop: 1, passedThroughStop: 0)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                }

                // Bottom action card
                VStack(spacing: 14) {
                    // One-line trail tagline (mockup commit 7c5ba6c —
                    // sits above the stats row and below the chrome).
                    Text(trail.summary)
                        .font(AppFont.sans(14.5, .regular))
                        .foregroundStyle(AppColor.ink100.opacity(0.92))
                        .multilineTextAlignment(.center)
                        .lineSpacing(2)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 4)

                    HStack(spacing: 10) {
                        statText(value: formattedMiles, suffix: "mi")
                        smallDot
                        statText(value: durationLabel.0, suffix: durationLabel.1)
                        smallDot
                        statText(value: "\(trail.stopCount)", suffix: "stops")
                        smallDot
                        Text(trail.difficulty)
                            .foregroundStyle(AppColor.ink80)
                    }
                    .font(AppFont.sans(13.5, .medium))

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

    private func statText(value: String, suffix: String) -> some View {
        HStack(spacing: 4) {
            Text(value)
                .font(AppFont.sans(13.5, .bold))
                .foregroundStyle(AppColor.ink100)
            Text(suffix)
                .foregroundStyle(AppColor.ink80)
        }
    }

    private var smallDot: some View {
        Circle()
            .frame(width: 3, height: 3)
            .foregroundStyle(AppColor.ink40)
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

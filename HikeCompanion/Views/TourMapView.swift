// TourMapView.swift
// In-tour full-screen map overlay. Tapped from the progress bar or stop
// hero in WalkingView. Same SwiftUI map as DetailView, but with the
// "passed" portion drawn solid lime up to the active stop.
//
// Mockup: design/mockups.html → .tour-map overlay.

import SwiftUI

struct TourMapView: View {
    let trail: Trail
    let activeStopIdx: Int      // 0-based
    let onClose: () -> Void

    private var activeStop: TrailStop? { trail.stops[safe: activeStopIdx] }
    private var activeStopNumber: Int { (activeStop?.number) ?? 1 }

    var body: some View {
        ZStack {
            AppColor.mapBg.ignoresSafeArea()

            VStack(spacing: 0) {
                // Top bar: close + title
                HStack(alignment: .center, spacing: 12) {
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.white)
                            .frame(width: 36, height: 36)
                            .background(.black.opacity(0.45), in: Circle())
                    }
                    .buttonStyle(.plain)

                    VStack(spacing: 2) {
                        Text(trail.name)
                            .font(AppFont.sans(16, .bold))
                            .tracking(-0.3)
                            .foregroundStyle(AppColor.ink100)
                        if let s = activeStop {
                            Text("Stop \(s.number) of \(trail.stops.count) · \(s.name)".uppercased())
                                .font(AppFont.sans(10, .heavy))
                                .tracking(1.8)
                                .foregroundStyle(AppColor.lime)
                        }
                    }
                    .frame(maxWidth: .infinity)

                    Color.clear.frame(width: 36, height: 36)
                }
                .padding(.horizontal, 18)
                .padding(.top, 60)
                .padding(.bottom, 8)

                // Map
                ZStack {
                    RadialGradient(
                        colors: [AppColor.lime.opacity(0.05), .clear],
                        center: .center,
                        startRadius: 20, endRadius: 240
                    )
                    // Edge-to-edge — no inset padding. The map should
                    // fill the canvas region between the top bar and
                    // the footer with no visible margin around it.
                    TrailMapView(trail: trail, activeStop: activeStopNumber, passedThroughStop: activeStopNumber)
                }

                // Footer
                HStack(spacing: 8) {
                    Circle()
                        .fill(AppColor.lime)
                        .frame(width: 6, height: 6)
                        .shadow(color: AppColor.lime.opacity(0.6), radius: 4)
                    Text(progressText)
                        .font(AppFont.sans(11, .heavy))
                        .tracking(1.6)
                        .textCase(.uppercase)
                        .foregroundStyle(AppColor.ink60)
                }
                .padding(.vertical, 14)
                .padding(.bottom, 30)
            }
        }
    }

    private var progressText: String {
        let walked = Double(activeStopIdx) / Double(max(1, trail.stops.count - 1)) * trail.distanceMiles
        return String(format: "%.1f of %.1f mi · in progress", walked, trail.distanceMiles)
    }
}

#Preview {
    TourMapView(trail: TrailData.kildoo, activeStopIdx: 2, onClose: {})
}

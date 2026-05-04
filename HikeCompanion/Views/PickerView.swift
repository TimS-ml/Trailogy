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
                    // Header
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Rectangle()
                                .frame(width: 16, height: 1)
                                .foregroundStyle(AppColor.lime.opacity(0.55))
                            Text("Nature, the best teacher.")
                                .font(AppFont.sans(10, .heavy))
                                .tracking(2.4)
                                .textCase(.uppercase)
                                .foregroundStyle(AppColor.lime)
                        }
                        .padding(.bottom, 18)

                        Text("Trails near you")
                            .font(AppFont.sans(28, .bold))
                            .foregroundStyle(AppColor.ink100)
                            .tracking(-0.5)

                        Text("Pittsburgh")
                            .font(AppFont.sans(13, .medium))
                            .foregroundStyle(AppColor.ink60)
                            .padding(.top, 4)
                    }
                    .padding(.horizontal, 22)
                    .padding(.top, 6)
                    .padding(.bottom, 22)

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
                .padding(.top, 70)  // status bar / dynamic island clearance
            }
            .scrollIndicators(.hidden)
        }
    }
}

// MARK: - TrailCard

private struct TrailCard: View {
    let trail: Trail
    let onTap: () -> Void

    var status: TrailStatus { TrailData.status(for: trail) }

    var body: some View {
        Button(action: onTap) {
            // Picture-on-top + text-panel-below.
            //
            // Why this shape (vs the previous overlay-with-scrim):
            //   - The image's native aspect (most Wikimedia hike photos
            //     are landscape ~3:2) doesn't match the old 16:13 card,
            //     so .scaledToFill cropped unpredictably and the bottom
            //     of the photo (and the text overlaid on it) was getting
            //     visually swallowed.
            //   - A dedicated text panel guarantees the region / trail
            //     name / stats line are always readable regardless of
            //     how the photo crops.
            //   - Layout is now deterministic per row: the image area
            //     is exactly 16:9 of the card width, the text panel sits
            //     beneath at its intrinsic content height.
            VStack(spacing: 0) {
                photoArea
                textPanel
            }
            .frame(maxWidth: .infinity)
            .background(AppColor.ink25)
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        }
        .buttonStyle(CardPressStyle())
    }

    /// Top of the card — the trail's cover photo at a 16:9 landscape
    /// aspect (matches typical Wikimedia hike photos so the crop is
    /// minimal). Status badge floats top-right.
    private var photoArea: some View {
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
        .frame(maxWidth: .infinity)
        .aspectRatio(16.0/9.0, contentMode: .fill)
        .clipped()
        .overlay(alignment: .topTrailing) {
            statusBadge
                .padding(14)
        }
    }

    /// Bottom of the card — solid dark text panel with region eyebrow,
    /// trail name, and the length / difficulty / time stats row.
    private var textPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(trail.region.uppercased())
                .font(AppFont.sans(10, .semibold))
                .tracking(1.6)
                .foregroundStyle(AppColor.ink60)
                .lineLimit(1)
                .minimumScaleFactor(0.85)

            Text(trail.name)
                .font(AppFont.sans(24, .bold))
                .foregroundStyle(AppColor.ink100)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 10) {
                Text("\(formattedMiles) mi")
                circleDot
                Text(trail.difficulty)
                circleDot
                Text(durationLabel)
            }
            .font(AppFont.sans(13, .medium))
            .foregroundStyle(AppColor.ink80)
            .padding(.top, 2)
            .lineLimit(1)
            .minimumScaleFactor(0.85)
        }
        .padding(.horizontal, 18)
        .padding(.top, 14)
        .padding(.bottom, 18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            // Slightly lifted from the page background so the panel
            // reads as a distinct surface under the photo.
            Color(red: 0.078, green: 0.082, blue: 0.071)  // ~ #14150f
        )
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

    @ViewBuilder
    private var statusBadge: some View {
        switch status {
        case .ready:
            Label {
                Text("Ready")
                    .font(AppFont.sans(10.5, .semibold))
                    .tracking(0.6)
                    .textCase(.uppercase)
            } icon: {
                Circle()
                    .fill(AppColor.lime)
                    .frame(width: 6, height: 6)
                    .shadow(color: AppColor.lime.opacity(0.6), radius: 4)
            }
            .foregroundStyle(AppColor.ink100)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.5), in: Capsule())

        case .walked(let date):
            Text(date)
                .font(AppFont.sans(10.5, .semibold))
                .tracking(0.6)
                .textCase(.uppercase)
                .foregroundStyle(AppColor.ink100.opacity(0.75))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.black.opacity(0.4), in: Capsule())

        case .downloadable:
            // Models are bundled — kept here as a placeholder if we ever
            // add real per-trail downloads.
            Label {
                Text("Download · \(byteLabel)")
                    .font(AppFont.sans(10.5, .heavy))
                    .tracking(0.6)
                    .textCase(.uppercase)
            } icon: {
                Image(systemName: "arrow.down.to.line")
                    .font(.system(size: 9, weight: .bold))
            }
            .foregroundStyle(AppColor.lime)
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(.black.opacity(0.55), in: Capsule())
            .overlay(
                Capsule().stroke(AppColor.lime.opacity(0.55), lineWidth: 1)
            )
        }
    }

    private var byteLabel: String {
        let mb = trail.bytes / (1024 * 1024)
        return "\(mb) MB"
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

// CachedTrailImage.swift
// Local-first image view for trail covers and stop photos.
//
// Drop-in replacement for `AsyncImage(url: ...) { phase in ... }` —
// the closure receives the same `AsyncImagePhase` so existing switch
// statements transplant unchanged. Tries `ImageStore.loadLocal` for
// the (trail, kind) first; falls back to AsyncImage over the network
// only if there's no local copy yet.
//
// USAGE
// -----
//   CachedTrailImage(trail: trail, kind: .cover) { phase in
//     switch phase {
//     case .success(let img): img.resizable().scaledToFill()
//     case .empty, .failure: AppColor.ink25
//     ...
//     }
//   }

import SwiftUI

struct CachedTrailImage<Content: View>: View {
    let trail: Trail
    let kind: TrailImageKind
    let content: (AsyncImagePhase) -> Content

    /// Cached UIImage for this (trail, kind). Resolved synchronously
    /// in `body` via ImageStore's in-memory cache; `.task(id:)`
    /// re-checks disk on appear / when the (trail, kind) identity
    /// changes, so images downloaded after this view first rendered
    /// take over without manual invalidation.
    @State private var localImage: UIImage?

    init(
        trail: Trail,
        kind: TrailImageKind,
        @ViewBuilder content: @escaping (AsyncImagePhase) -> Content
    ) {
        self.trail = trail
        self.kind = kind
        self.content = content
    }

    var body: some View {
        Group {
            if let img = localImage ?? ImageStore.loadLocal(for: trail, kind: kind) {
                content(.success(Image(uiImage: img)))
            } else if let url = remoteURL {
                // No local copy yet — fall back to a network fetch.
                // AsyncImage's URLCache may have something even if our
                // disk doesn't (e.g. user is on Wi-Fi but hasn't tapped
                // Download for this trail yet).
                AsyncImage(url: url, content: content)
            } else {
                content(.empty)
            }
        }
        .task(id: identityKey) {
            // Pick up freshly-downloaded files. Re-firing on
            // (trail.id, kind) change is safe because the trailing
            // .task wires `id:` to a re-run.
            localImage = ImageStore.loadLocal(for: trail, kind: kind)
        }
    }

    private var remoteURL: URL? {
        switch kind {
        case .cover:
            return trail.coverImageURL
        case .stop(let n):
            return trail.stops.first(where: { $0.number == n })?.imageURL
        }
    }

    /// Stable identity for `.task(id:)` — when this changes the
    /// task body re-runs and we re-check disk for an updated
    /// local copy.
    private var identityKey: String {
        switch kind {
        case .cover:       return "\(trail.id)::cover"
        case .stop(let n): return "\(trail.id)::stop-\(n)"
        }
    }
}

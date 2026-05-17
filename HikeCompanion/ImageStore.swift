// ImageStore.swift
// On-disk image cache for offline trail viewing.
//
// PROBLEM
// -------
// AsyncImage fetches Wikimedia URLs at display time. On a trail in a
// canyon with no service, every cover/stop image fails to load and
// the UI shows blank rectangles. iOS's URLCache is best-effort —
// small, evictable, doesn't survive reinstall — so we can't rely on
// it for offline guarantees.
//
// SOLUTION
// --------
// When the user taps the detail-view's Download CTA, we explicitly
// fetch every image the trail will ever display (cover + each stop)
// and write them to Application Support. CachedTrailImage then loads
// from disk first and only falls back to AsyncImage if the local
// copy is missing.
//
// DISK LAYOUT
// -----------
//   {Application Support}/TrailImages/{trail.id}/cover.bin
//   {Application Support}/TrailImages/{trail.id}/stop-1.bin
//   ...stop-N.bin
//
// Files saved as raw bytes (.bin) regardless of format (JPEG / PNG /
// WebP) — UIImage(data:) handles all three. Per-trail subdirs make
// future per-trail delete trivial.

import Foundation
import UIKit

/// Identifies which image of a trail we're talking about. `cover`
/// drives the picker card + recap header; `.stop(N)` drives the
/// in-tour hero card for that stop (`stop.number` is 1-based).
enum TrailImageKind: Equatable {
    case cover
    case stop(Int)

    fileprivate var filename: String {
        switch self {
        case .cover:       return "cover.bin"
        case .stop(let n): return "stop-\(n).bin"
        }
    }
}

@MainActor
enum ImageStore {

    // MARK: - In-memory cache

    /// Decoded UIImage cache keyed by trail-id + kind. Strong refs
    /// (not NSCache) — the working set is small (~18 images across
    /// the three trails) and we want hot card-scroll re-renders to
    /// be instant. Evicted explicitly by `deleteAll(for:)` and on
    /// post-download writes.
    private static var memCache: [String: UIImage] = [:]

    private static func cacheKey(_ trail: Trail, _ kind: TrailImageKind) -> String {
        "\(trail.id)::\(kind.filename)"
    }

    // MARK: - Path resolution (nonisolated — pure I/O)

    /// Application Support / TrailImages — the canonical root.
    /// Created on first access. Using Application Support (not
    /// Caches) because these files are user-meaningful and shouldn't
    /// be evicted by the system under memory pressure.
    nonisolated static var rootDir: URL {
        let fm = FileManager.default
        let base = try? fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let root = (base ?? URL(fileURLWithPath: NSTemporaryDirectory()))
            .appendingPathComponent("TrailImages", isDirectory: true)
        try? fm.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    nonisolated static func localURL(for trail: Trail, kind: TrailImageKind) -> URL {
        rootDir
            .appendingPathComponent(trail.id, isDirectory: true)
            .appendingPathComponent(kind.filename, isDirectory: false)
    }

    /// True iff every image the trail will display is on disk.
    /// Drives `AppRouter.isDownloaded`'s disk-backed seed.
    nonisolated static func hasAllLocal(for trail: Trail) -> Bool {
        let fm = FileManager.default
        let coverURL = localURL(for: trail, kind: .cover)
        guard fm.fileExists(atPath: coverURL.path) else { return false }
        for stop in trail.stops where stop.imageURL != nil {
            let stopURL = localURL(for: trail, kind: .stop(stop.number))
            if !fm.fileExists(atPath: stopURL.path) { return false }
        }
        return true
    }

    // MARK: - Local lookup

    /// Returns the decoded UIImage if it's on disk. Caches in memory
    /// so subsequent reads skip the disk + decode. nil = not present
    /// (file missing or empty), caller should fall back to network.
    static func loadLocal(for trail: Trail, kind: TrailImageKind) -> UIImage? {
        let key = cacheKey(trail, kind)
        if let cached = memCache[key] { return cached }

        let url = localURL(for: trail, kind: kind)
        guard let data = try? Data(contentsOf: url), !data.isEmpty,
              let img = UIImage(data: data) else {
            return nil
        }
        memCache[key] = img
        return img
    }

    // MARK: - Download

    /// Download every image the trail uses (cover + each stop) and
    /// write atomically to local disk. Calls `progress(0...1)` as
    /// images complete — drives the CTA's real progress bar.
    ///
    /// Per-image failure logs and continues — a partial trail
    /// (cover ok, one stop image broken) is still useful. The trail
    /// counts as fully downloaded only when `hasAllLocal` returns
    /// true afterwards.
    static func downloadAll(
        for trail: Trail,
        progress: @escaping @MainActor (Double) -> Void
    ) async throws {
        var tasks: [(kind: TrailImageKind, url: URL)] = []
        if let url = trail.coverImageURL {
            tasks.append((.cover, url))
        }
        for stop in trail.stops {
            if let url = stop.imageURL {
                tasks.append((.stop(stop.number), url))
            }
        }
        let total = max(1, tasks.count)
        await progress(0)

        let trailDir = rootDir.appendingPathComponent(trail.id, isDirectory: true)
        try FileManager.default.createDirectory(at: trailDir, withIntermediateDirectories: true)

        for (i, task) in tasks.enumerated() {
            do {
                let (data, response) = try await URLSession.shared.data(from: task.url)
                if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                    print("[ImageStore] HTTP \(http.statusCode) for \(task.url.lastPathComponent), skipping")
                } else {
                    let dest = localURL(for: trail, kind: task.kind)
                    try data.write(to: dest, options: .atomic)
                    // Drop any stale in-memory cache entry so the next
                    // read picks up the freshly-written bytes.
                    memCache.removeValue(forKey: cacheKey(trail, task.kind))
                }
            } catch {
                print("[ImageStore] download failed for \(task.url): \(error.localizedDescription)")
            }
            await progress(Double(i + 1) / Double(total))
        }
    }

    /// Delete every cached image for a trail. Not exposed in UI
    /// yet — here for future per-trail eviction support.
    static func deleteAll(for trail: Trail) {
        let trailDir = rootDir.appendingPathComponent(trail.id, isDirectory: true)
        try? FileManager.default.removeItem(at: trailDir)
        let prefix = trail.id + "::"
        memCache = memCache.filter { !$0.key.hasPrefix(prefix) }
    }
}

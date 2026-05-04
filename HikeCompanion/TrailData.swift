// TrailData.swift
// Static sample data taken from design/mockups.html. The mockup demonstrates
// a "Kildoo Trail" tour at McConnells Mill State Park with 5 stops; we keep
// the same content so the implementation matches the design 1-to-1.
//
// The trail/stop *imagery* in the mockup uses Wikimedia URLs; we reference
// those by URL too — SwiftUI's `AsyncImage` will fetch them at runtime.
// (This is fine for the prototype; pre-bundle the photos later if you want
// the app to work fully offline at the trailhead.)
//
// Real GPS positioning, real downloads, and real per-trail content all
// require infrastructure we don't have yet. For now the data is a frame.

import Foundation

struct TrailStop: Identifiable, Hashable {
    let id = UUID()
    let number: Int
    let name: String
    let imageURL: URL?
    /// Pre-written narration sentences for this stop. Read aloud in
    /// sequence by the lyric loop. (Could be Gemma-generated per-stop in
    /// the future, but generating on-device per arrival is expensive —
    /// pre-writing is more reliable and faster.)
    let sentences: [String]
    /// One-line factual summary used in the post-tour journal entry.
    let journalFact: String
}

struct Trail: Identifiable, Hashable {
    let id: String
    let name: String
    let region: String
    let parkLocation: String
    let distanceMiles: Double
    let durationMinutes: Int
    let difficulty: String
    let stopCount: Int
    let bytes: Int                  // bundle size for download flow (mockup-only)
    let coverImageURL: URL?
    let stops: [TrailStop]

    /// Distance + walking-time labels for the segment from `stops[i]` to
    /// `stops[i+1]`. Mockup-only string content.
    let segmentLabels: [String]

    /// 0..1 horizontal position of each stop on the progress bar. Five
    /// equally-spaced positions for Kildoo (12, 32, 50, 70, 88 in the mockup).
    let stopProgressPositions: [Double]

    /// Spoken when the user taps "Begin". Kokoro reads this as a
    /// trail-guide-style introduction. Length is tuned so reading takes
    /// roughly 30–60 s — long enough to feel substantial, short enough
    /// to keep the user engaged before they want to interact.
    /// The tour-cycle phase timer pauses while this is speaking.
    let intro: String
}

extension TrailStop {
    /// All sentences joined with single spaces — convenience for feeding
    /// the whole stop blurb to Kokoro in one synth call.
    var spokenNarration: String {
        sentences.joined(separator: " ")
    }
}

enum TrailStatus: Equatable {
    case ready          // bundled / already downloaded
    case downloadable(downloadedFraction: Double = 0)
    case walked(dateLabel: String)
}

// MARK: - Sample data

enum TrailData {

    static let kildoo = Trail(
        id: "kildoo",
        name: "Kildoo Trail",
        region: "McConnells Mill",
        parkLocation: "McConnells Mill State Park",
        distanceMiles: 2.0,
        durationMinutes: 60,
        difficulty: "Moderate",
        stopCount: 5,
        bytes: 68 * 1_024 * 1_024,
        coverImageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/c/cc/Woodlands_around_McConnell%27s_Mill_State_Park.jpg"),
        stops: [
            TrailStop(
                number: 1,
                name: "Covered Bridge & Mill",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/7/77/McConnells_Mill_State_Park_Bridge.jpg"),
                sentences: [
                    "The covered bridge dates to 1874.",
                    "Howe truss design — one of two left in Pennsylvania.",
                    "The mill ground grain here until 1928."
                ],
                journalFact: "Built in 1874 — one of two Howe-truss bridges left in Pennsylvania. The mill ground grain here until 1928."
            ),
            TrailStop(
                number: 2,
                name: "Layered Cliffs",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/c/cc/Woodlands_around_McConnell%27s_Mill_State_Park.jpg"),
                sentences: [
                    "Sandstone laid down 320 million years ago.",
                    "The orange streaks are iron oxide.",
                    "Groundwater carries it out of the rock."
                ],
                journalFact: "Sandstone laid down 320 million years ago — the orange streaks are iron oxide leached from the rock by groundwater."
            ),
            TrailStop(
                number: 3,
                name: "Kildoo Falls",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/2/22/Hell%27s_Hollow_Falls.JPG"),
                sentences: [
                    "The water has been falling here a long time.",
                    "Listen. The creek does most of the work.",
                    "Above you, hemlocks lean over the gorge.",
                    "Some of them are three centuries old."
                ],
                journalFact: "The eastern hemlocks above the gorge are roughly three centuries old — older than the country."
            ),
            TrailStop(
                number: 4,
                name: "Eckert Bridge",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/3/34/McConnells_Mill_Bridge_and_Creek.jpg"),
                sentences: [
                    "Eckert Bridge crosses Slippery Rock Creek here.",
                    "The trail returns north along the western bank.",
                    "The creek narrows and quickens through the gorge."
                ],
                journalFact: "South crossing back to the western bank. The creek narrows here — a good spot to pause."
            ),
            TrailStop(
                number: 5,
                name: "The Slippery Rock",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/2/26/McConnells_Mill_State_Park_Scenery_01.jpg"),
                sentences: [
                    "An eighty-ton sandstone boulder in the creek.",
                    "It gave the waterway its name.",
                    "Algae makes it slick — hence slippery."
                ],
                journalFact: "The 80-ton sandstone boulder in the creek that gave the waterway its name."
            )
        ],
        segmentLabels: [
            "0.4 mi · about 12 minutes",
            "0.4 mi · about 11 minutes",
            "0.3 mi · about 9 minutes",
            "0.5 mi · about 14 minutes",
            "0.4 mi · back to the mill"
        ],
        stopProgressPositions: [0.12, 0.32, 0.50, 0.70, 0.88],
        intro: """
        Welcome to the Kildoo Trail. This is a two-mile loop rated \
        moderate — about an hour at a comfortable pace, with some \
        uneven footing where we drop into the gorge. We're at \
        McConnells Mill State Park, on the west bank of Slippery Rock \
        Creek. The creek carved this sandstone gorge about fifteen \
        thousand years ago, when the last glaciers retreated and a \
        melt-water torrent cut this channel in a few hundred years.
        We'll pass a covered bridge from 1874 — one of two Howe-truss \
        bridges left in Pennsylvania — and a four-story grist mill that \
        ground grain here until 1928. Above the gorge, eastern hemlocks \
        lean over the water; some of them are three centuries old, \
        older than the country itself.
        Take your time. Listen for the creek. Stop and ask me about \
        anything you see along the way — a tree, a rock, a bird call. \
        I'll be right here when you're ready.
        """
    )

    static let hellsHollow = Trail(
        id: "hells",
        name: "Hells Hollow",
        region: "McConnells Mill",
        parkLocation: "McConnells Mill State Park",
        distanceMiles: 1.2,
        durationMinutes: 50,
        difficulty: "Easy",
        stopCount: 3,
        bytes: 41 * 1_024 * 1_024,
        coverImageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/2/22/Hell%27s_Hollow_Falls.JPG"),
        stops: [
            TrailStop(
                number: 1,
                name: "Trailhead Steps",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/2/22/Hell%27s_Hollow_Falls.JPG"),
                sentences: [
                    "The trail drops sharply down a wooded ravine.",
                    "Keep an eye on the railings — the steps stay slick year-round.",
                    "Hemlock and rhododendron close in overhead."
                ],
                journalFact: "The descent into the gorge passes under a dense hemlock canopy."
            ),
            TrailStop(
                number: 2,
                name: "Hells Hollow Falls",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/2/22/Hell%27s_Hollow_Falls.JPG"),
                sentences: [
                    "The falls cut through a thick limestone shelf.",
                    "The cool air at the base is the same year-round.",
                    "This is the sound the place is known for."
                ],
                journalFact: "Hells Hollow Falls drops over a limestone bed laid down 350 million years ago."
            ),
            TrailStop(
                number: 3,
                name: "Creek Junction",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/2/26/McConnells_Mill_State_Park_Scenery_01.jpg"),
                sentences: [
                    "Hells Run meets Slippery Rock Creek here.",
                    "From here the trail loops back to the trailhead.",
                    "Listen for wood thrush in spring."
                ],
                journalFact: "Hells Run flows into Slippery Rock Creek at this junction — a quiet spot to rest before climbing back out."
            )
        ],
        segmentLabels: [
            "0.5 mi · about 14 minutes",
            "0.4 mi · about 11 minutes",
            "0.3 mi · back to trailhead"
        ],
        stopProgressPositions: [0.18, 0.50, 0.82],
        intro: """
        Welcome to Hells Hollow. This is a short, easy loop on the \
        north end of McConnells Mill State Park — about a mile and a \
        quarter, fifty minutes round trip. The trail drops a hundred \
        and fifty feet over half a mile, then levels out at a limestone \
        waterfall. The descent is shaded by eastern hemlock and great \
        rhododendron the whole way down — stay close to the railings, \
        the steps stay damp year-round. Once we reach the falls, \
        listen. The sound is what this place is known for.
        """
    )

    static let tranquil = Trail(
        id: "tranquil",
        name: "Tranquil Trail",
        region: "Frick Park",
        parkLocation: "Frick Park",
        distanceMiles: 1.5,
        durationMinutes: 45,
        difficulty: "Easy",
        stopCount: 4,
        bytes: 52 * 1_024 * 1_024,
        coverImageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
        stops: [
            TrailStop(
                number: 1,
                name: "Bridge Trail Entrance",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
                sentences: [
                    "Frick Park is the largest of Pittsburgh's regional parks.",
                    "The trail follows an old streetcar grade.",
                    "Oaks here are over a hundred years old."
                ],
                journalFact: "Frick Park's 644 acres were donated by Henry Clay Frick's daughter in 1919."
            ),
            TrailStop(
                number: 2,
                name: "Nine Mile Run",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
                sentences: [
                    "Nine Mile Run flows through the park to the Monongahela.",
                    "The stream was once buried under industrial slag.",
                    "Restoration work brought it back to daylight in 2006."
                ],
                journalFact: "Nine Mile Run was uncovered in 2006 — one of the largest urban stream restorations in the U.S."
            ),
            TrailStop(
                number: 3,
                name: "Falls Ravine",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
                sentences: [
                    "A small seasonal waterfall sits just off the trail.",
                    "Most of the year, only moss-covered stones mark the spot.",
                    "After rain, the falls run again for a day or two."
                ],
                journalFact: "An ephemeral waterfall — visible only after heavy rain."
            ),
            TrailStop(
                number: 4,
                name: "Forbes Overlook",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
                sentences: [
                    "From here you can see across the Monongahela to Squirrel Hill.",
                    "The light is best in late afternoon.",
                    "From here it's a short climb back to Forbes Avenue."
                ],
                journalFact: "Overlook from the southern edge of Frick Park, just before the trail returns to Forbes Avenue."
            )
        ],
        segmentLabels: [
            "0.4 mi · about 4 minutes",
            "0.4 mi · about 4 minutes",
            "0.4 mi · about 4 minutes",
            "0.3 mi · back to start"
        ],
        stopProgressPositions: [0.12, 0.38, 0.62, 0.88],
        intro: """
        Welcome to the Tranquil Trail in Pittsburgh's Frick Park. \
        This is one of the gentlest walks in the city — a mile and a \
        half, rated easy, about forty-five minutes at a comfortable \
        pace. The trail follows an old streetcar grade through a \
        hardwood forest of oak, beech, and tulip poplar. Some of these \
        trees are over a hundred years old. The path runs alongside \
        Nine Mile Run, a stream that was buried under industrial slag \
        for most of the twentieth century and brought back to daylight \
        in 2006. Take your time and enjoy the quiet — you're inside \
        one of the few stretches of urban old growth left in Pennsylvania.
        """
    )

    /// Order shown on the picker.
    static let all: [Trail] = [kildoo, hellsHollow, tranquil]

    static func status(for trail: Trail) -> TrailStatus {
        switch trail.id {
        case "hells":   return .walked(dateLabel: "Apr 14")
        // Models are bundled at app install — every trail is "ready" from
        // the user's perspective; the per-trail download flow in the
        // mockup is decorative.
        default:        return .ready
        }
    }
}

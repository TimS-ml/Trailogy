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
    /// What to notice on the WALK to the next stop — surfaced in the
    /// "ON THE WAY" panel during between/approaching phases. The
    /// engagement-loop half: invites the user to look for something
    /// specific on the trail between stops. Nil on the last stop of
    /// a trail (no next stop to walk to). See design/README.md
    /// "Look-for / payoff engagement loop" for the design rationale.
    let lookFor: String?
    /// Callback that resolves the PREVIOUS stop's `lookFor` — prepended
    /// to the spoken narration on arrival ("If you saw a long
    /// stone-lined trench…"). Nil on the first stop of a trail (no
    /// previous lookFor to resolve).
    let payoff: String?

    init(
        number: Int,
        name: String,
        imageURL: URL?,
        sentences: [String],
        journalFact: String,
        lookFor: String? = nil,
        payoff: String? = nil
    ) {
        self.number = number
        self.name = name
        self.imageURL = imageURL
        self.sentences = sentences
        self.journalFact = journalFact
        self.lookFor = lookFor
        self.payoff = payoff
    }
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

    /// Domain priors fed to Gemma as part of the system prompt so it
    /// can ground answers in the actual ecosystem of THIS trail rather
    /// than answering as a generic outdoor companion. Should cover the
    /// trail's common flora, fauna, and geology in ~80–120 words.
    /// Without this the model knows nothing about the region — it'll
    /// hallucinate species or hedge so hard the answer is useless.
    let regionalContext: String
}

extension TrailStop {
    /// All sentences joined with single spaces — convenience for feeding
    /// the whole stop blurb to Kokoro in one synth call.
    var spokenNarration: String {
        sentences.joined(separator: " ")
    }

    /// Spoken narration with the optional `payoff` prepended as the
    /// first sentence. Used when the user arrives at a stop after a
    /// walk where a `lookFor` was suggested — the payoff is the
    /// callback resolution, then the regular sentences continue. If
    /// no payoff is set (e.g. stop 1 of a trail), this is identical
    /// to `spokenNarration`.
    var spokenNarrationWithPayoff: String {
        if let payoff, !payoff.isEmpty {
            return ([payoff] + sentences).joined(separator: " ")
        }
        return spokenNarration
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
                journalFact: "Built in 1874 — one of two Howe-truss bridges left in Pennsylvania. The mill ground grain here until 1928.",
                lookFor: "Look for the mill race — a stone-lined channel along the creek that fed the wheel."
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
                journalFact: "Sandstone laid down 320 million years ago — the orange streaks are iron oxide leached from the rock by groundwater.",
                lookFor: "Listen for the falls. You'll hear them before you see them.",
                payoff: "If you saw a long stone-lined trench along the creek, that was the old mill race."
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
                journalFact: "The eastern hemlocks above the gorge are roughly three centuries old — older than the country.",
                lookFor: "Watch the hemlocks for buck rubs — bark scraped smooth at chest height.",
                payoff: "You probably heard the falls a quarter-mile back — water travels far in a gorge."
            ),
            // Stops 4 and 5 swapped to match design/mockups.html order
            // (Slippery Rock at 4, Eckert Bridge at 5). The mockup's
            // lookFor / payoff arc is tuned for this order: stop 3 →
            // 4 is about buck rubs on hemlocks; stop 4 → 5 is about
            // polypody fern rooted in bare stone.
            TrailStop(
                number: 4,
                name: "Slippery Rock",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/2/26/McConnells_Mill_State_Park_Scenery_01.jpg"),
                sentences: [
                    "An eighty-ton sandstone boulder in the creek.",
                    "It gave the waterway its name.",
                    "Algae makes it slick — hence slippery."
                ],
                journalFact: "The 80-ton sandstone boulder in the creek that gave the waterway its name.",
                lookFor: "Find a fern rooted in bare stone, no soil.",
                payoff: "Those bark scrapes are bucks marking territory in fall."
            ),
            TrailStop(
                number: 5,
                name: "Eckert Bridge",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/3/34/McConnells_Mill_Bridge_and_Creek.jpg"),
                sentences: [
                    "Eckert Bridge crosses Slippery Rock Creek here.",
                    "The trail returns north along the western bank.",
                    "The creek narrows and quickens through the gorge."
                ],
                journalFact: "South crossing back to the western bank. The creek narrows here — a good spot to pause.",
                payoff: "That's polypody fern — it roots in moss on bare rock and curls up to survive drought."
            )
        ],
        // Segment distances mirror the mockup's TRAILS.kildoo.segmentDistances
        // verbatim. Each entry covers the walk FROM stops[i] to stops[i+1]
        // (and segment[count-1] is the closing leg back to the mill).
        segmentLabels: [
            "0.5 mi · about 13 minutes",
            "0.4 mi · about 12 minutes",
            "0.5 mi · about 14 minutes",
            "0.4 mi · about 11 minutes",
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
        """,
        regionalContext: """
        Trees: eastern hemlock, white oak, red oak, sugar maple, \
        beech, tulip poplar, sassafras, rhododendron. Wildlife: \
        white-tailed deer, pileated woodpeckers, red-tailed hawks, \
        brook trout. Pennsylvanian sandstone gorge (~320 Mya), carved \
        by glacial meltwater ~15,000 years ago. Iron-oxide staining \
        from groundwater seeps; mosses and ferns at the creek.
        """
    )

    /// Old Field & Jennings Trail Loop at the Wildflower Reserve,
    /// Raccoon Creek State Park. Replaces the previous Hells Hollow
    /// trail (the design/mockups.html source-of-truth swapped this in,
    /// see design/README.md "Hells Hollow → Wildflower Reserve" entry).
    /// Path is stitched from real OSM ways (Old Field Trail [Red] +
    /// Jennings Trail [Blue]) — GPX-accurate, unlike Kildoo/Tranquil
    /// whose coordinates are visual estimates.
    static let oldField = Trail(
        id: "oldfield",
        name: "Old Field & Jennings",
        region: "Wildflower Reserve",
        parkLocation: "Raccoon Creek State Park",
        distanceMiles: 2.3,
        durationMinutes: 50,
        difficulty: "Easy",
        stopCount: 5,
        bytes: 41 * 1_024 * 1_024,
        coverImageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/b/b6/Jennings_Environmental_Education_Center.jpg"),
        stops: [
            TrailStop(
                number: 1,
                name: "Trailhead",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/b/b6/Jennings_Environmental_Education_Center.jpg"),
                sentences: [
                    "You're at the southwest entrance to the Wildflower Reserve loop.",
                    "These woods hold one of the richest spring wildflower displays in Pennsylvania.",
                    "Trilliums, Virginia bluebells, and Dutchman's breeches all bloom here."
                ],
                journalFact: "The southwest trailhead at the Wildflower Reserve — gateway to one of Pennsylvania's richest spring ephemeral displays.",
                lookFor: "Look for trillium on the forest floor — three white petals on a single stem."
            ),
            TrailStop(
                number: 2,
                name: "Wildflower Meadow",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
                sentences: [
                    "A sunlit clearing along the Old Field Trail.",
                    "In late April, the forest floor here turns white with large-flowered trillium.",
                    "Wood thrushes and ovenbirds sing from the surrounding canopy."
                ],
                journalFact: "Wildflower Meadow — large-flowered trillium carpets the forest floor in late April.",
                lookFor: "Look for old fence posts in the woods. These slopes used to be farmland.",
                payoff: "If you spotted three white petals close to the ground, you found trillium — it takes seven years to bloom from seed."
            ),
            TrailStop(
                number: 3,
                name: "East Overlook",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/b/b6/Jennings_Environmental_Education_Center.jpg"),
                sentences: [
                    "The eastern turn of the loop, where Old Field meets Jennings Trail.",
                    "The slope below drops down toward Raccoon Creek.",
                    "On still mornings, mist rises through the trees in the valley."
                ],
                journalFact: "East Overlook — the slope drops toward Raccoon Creek; Old Field and Jennings join here.",
                lookFor: "On the way down, count the fallen trees left across the slope.",
                payoff: "Those weathered posts marked the edge of an 1800s farm — the forest reclaimed it."
            ),
            TrailStop(
                number: 4,
                name: "Forest Glen",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/c/cc/Woodlands_around_McConnell%27s_Mill_State_Park.jpg"),
                sentences: [
                    "Deep into the Jennings Trail, the canopy thickens overhead.",
                    "Look for jack-in-the-pulpit and wild ginger close to the path.",
                    "Pileated woodpeckers work the dead snags here year-round."
                ],
                journalFact: "Forest Glen — deep canopy along Jennings Trail; jack-in-the-pulpit and pileated woodpeckers.",
                lookFor: "Listen for woodpeckers drumming on the dead snags.",
                payoff: "Fallen trees here are left in place — they're nurseries for moss, fungi, and new saplings."
            ),
            TrailStop(
                number: 5,
                name: "Loop Close",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/c/cc/Woodlands_around_McConnell%27s_Mill_State_Park.jpg"),
                sentences: [
                    "The last stretch back toward the trailhead.",
                    "The understory opens up — sugar maple, beech, and red oak.",
                    "A few quiet minutes and the loop is closed."
                ],
                journalFact: "The loop closes — sugar maple, beech, and red oak see us back to the start.",
                payoff: "That tapping was likely a pileated woodpecker — the largest in eastern forests."
            )
        ],
        segmentLabels: [
            "0.3 mi · about 7 minutes",
            "0.4 mi · about 9 minutes",
            "0.9 mi · about 20 minutes",
            "0.6 mi · about 12 minutes",
            "0.1 mi · final stretch"
        ],
        stopProgressPositions: [0.12, 0.31, 0.50, 0.69, 0.88],
        intro: """
        Welcome to the Old Field and Jennings Trail Loop. This is a \
        two-point-three mile loop at the Wildflower Reserve, in the \
        northern end of Raccoon Creek State Park — about fifty minutes \
        at a comfortable pace, rated easy. We're walking through one \
        of the richest spring wildflower displays in Pennsylvania. \
        In late April and early May, the forest floor here turns white \
        with large-flowered trillium, blue with Virginia bluebells, \
        and dotted with the pale lanterns of Dutchman's breeches.
        We'll start at the southwest trailhead, head east through old \
        fields and meadow edges, then return through deeper forest \
        along the Jennings Trail back to where we began. Listen for \
        wood thrush and ovenbird in spring.
        Stop and ask me about any flower, tree, or bird call along \
        the way. I'll be right here when you're ready.
        """,
        regionalContext: """
        Trees: sugar maple, American beech, red oak, white oak, tulip \
        poplar, black cherry, hickory. Spring wildflowers (April-May): \
        large-flowered trillium, Virginia bluebells, Dutchman's \
        breeches, jack-in-the-pulpit, wild ginger, spring beauty, \
        trout lily. Wildlife: white-tailed deer, wild turkey, pileated \
        woodpeckers, wood thrush (spring), ovenbird, ruffed grouse, \
        red foxes. Pennsylvanian sandstone and shale; Raccoon Creek \
        watershed; one of the richest spring ephemeral wildflower \
        displays in western Pennsylvania.
        """
    )

    /// Tranquil Trail in Frick Park. Realigned with the mockup
    /// source-of-truth (design/mockups.html → `TRAILS.tranquil`):
    /// 1.1 mi out-and-back (was 1.5 mi loop), 3 stops (was 4), 30 min
    /// (was 45). Stops are now Trailhead / Fern Hollow Creek / Forest
    /// Grove — completely different from the previous Bridge / Nine
    /// Mile Run / Falls Ravine / Forbes Overlook set.
    static let tranquil = Trail(
        id: "tranquil",
        name: "Tranquil Trail",
        region: "Frick Park",
        parkLocation: "Frick Park",
        distanceMiles: 1.1,
        durationMinutes: 30,
        difficulty: "Easy",
        stopCount: 3,
        bytes: 52 * 1_024 * 1_024,
        coverImageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
        stops: [
            TrailStop(
                number: 1,
                name: "Trailhead",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
                sentences: [
                    "The trail begins off Beechwood Boulevard.",
                    "Frick Park is Pittsburgh's largest historic park.",
                    "The forest here regrew over a century after logging."
                ],
                journalFact: "The trail begins off Beechwood Boulevard. Frick Park is Pittsburgh's largest historic park — 644 acres donated by Helen Clay Frick in 1919.",
                lookFor: "Look for skunk cabbage in the wet spots — broad green hoods close to the ground."
            ),
            TrailStop(
                number: 2,
                name: "Fern Hollow Creek",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/6/60/Pittsburgh_frick_park_trail.jpg"),
                sentences: [
                    "Fern Hollow Creek runs alongside the trail.",
                    "It empties into Nine Mile Run downstream.",
                    "The valley was carved during the last glacial period."
                ],
                journalFact: "Fern Hollow Creek runs alongside the trail and empties into Nine Mile Run downstream (daylit and restored in 2006).",
                lookFor: "Look for old streetcar bricks along the creek bank.",
                payoff: "Skunk cabbage is the first plant to bloom each year — it can melt its own snow with metabolic heat."
            ),
            TrailStop(
                number: 3,
                name: "Forest Grove",
                imageURL: URL(string: "https://upload.wikimedia.org/wikipedia/commons/c/cc/Woodlands_around_McConnell%27s_Mill_State_Park.jpg"),
                sentences: [
                    "Mature oaks and tulip poplars stand here.",
                    "Some of these trees are over 150 years old.",
                    "The understory is mostly spicebush and witch hazel."
                ],
                journalFact: "Mature oaks and tulip poplars — some over 150 years old. Turnaround point of this out-and-back walk.",
                payoff: "Those red bricks are from Pittsburgh's old streetcar lines — many washed downstream over the decades."
            )
        ],
        segmentLabels: [
            "0.3 mi · about 8 minutes",
            "0.3 mi · about 7 minutes",
            "— turn around at the grove"
        ],
        stopProgressPositions: [0.12, 0.50, 0.88],
        intro: """
        Welcome to the Tranquil Trail in Frick Park — Pittsburgh's \
        largest historic park. This is a short out-and-back walk: a \
        mile and a tenth from start to turnaround and back, about \
        thirty minutes at a comfortable pace, rated easy. The trail \
        begins off Beechwood Boulevard and drops gently into the \
        valley of Fern Hollow Creek, which empties into Nine Mile \
        Run downstream. The forest here regrew over a century after \
        logging, and the path passes through mature stands of oak \
        and tulip poplar — some of these trees are over a hundred \
        and fifty years old. Listen for thrushes and woodpeckers, \
        watch for spicebush and witch hazel in the understory, and \
        take your time. We'll turn around at Forest Grove.
        """,
        regionalContext: """
        Trees: white oak, red oak, American beech, tulip poplar, \
        sugar maple, black cherry, spicebush and witch hazel \
        (understory). Wildlife: gray squirrels, white-tailed deer, \
        red-tailed hawks, barred owls, wood thrush, pileated \
        woodpeckers, red foxes (occasional). Fern Hollow Creek runs \
        alongside the trail and empties into Nine Mile Run \
        downstream — a daylit urban stream restored from 2002. \
        Pittsburgh Coal strata (Pennsylvanian, ~300 Mya); the \
        valley was carved during the last glacial period.
        """
    )

    /// Order shown on the picker.
    static let all: [Trail] = [kildoo, oldField, tranquil]

    static func status(for trail: Trail) -> TrailStatus {
        switch trail.id {
        // Old Field & Jennings shows as "Completed Apr 14" on the
        // picker — matches the mockup's per-card status badge.
        case "oldfield": return .walked(dateLabel: "Apr 14")
        // Models are bundled at app install — every trail is "ready" from
        // the user's perspective; the per-trail download flow in the
        // mockup is decorative.
        default:         return .ready
        }
    }
}

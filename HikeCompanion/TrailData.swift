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

import CoreLocation
import Foundation

/// Compact constructor for the dense coordinate-literal arrays below.
/// Keeps each path entry readable as `c(40.95, -80.17)` instead of the
/// noisier `CLLocationCoordinate2D(latitude: ..., longitude: ...)`.
private func c(_ lat: Double, _ lng: Double) -> CLLocationCoordinate2D {
    CLLocationCoordinate2D(latitude: lat, longitude: lng)
}

/// A single recap "Discovery" card — the post-tour learning anchored
/// by a hero number / date / quantity that makes the fact memorable.
/// Mirrors design/mockups.html `.learning-card` (anchor + body).
/// Curator-authored, static per trail — selective, not exhaustive
/// (not every stop produces a learning, and one stop can produce
/// more than one).
struct Learning: Identifiable {
    let id = UUID()
    /// Short heading — the hero number/date/quantity. Examples:
    /// "320 million years", "Iron oxide", "1874", "80 tons".
    let anchor: String
    /// One-paragraph context that unpacks the anchor.
    let body: String
}

struct TrailStop: Identifiable {
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
    /// Geographic position of this stop. Used by TrailMapView to drop
    /// the stop's annotation on the map. For Kildoo and Tranquil these
    /// are visual estimates aligned to the mockup's TRAILS data; for
    /// Old Field they are OSM-sourced vertices on the actual trail
    /// polyline (see Trail.path comment).
    let coordinate: CLLocationCoordinate2D

    init(
        number: Int,
        name: String,
        imageURL: URL?,
        sentences: [String],
        journalFact: String,
        coordinate: CLLocationCoordinate2D,
        lookFor: String? = nil,
        payoff: String? = nil
    ) {
        self.number = number
        self.name = name
        self.imageURL = imageURL
        self.sentences = sentences
        self.journalFact = journalFact
        self.coordinate = coordinate
        self.lookFor = lookFor
        self.payoff = payoff
    }
}

struct Trail: Identifiable {
    let id: String
    let name: String
    /// One-line tagline displayed under the trail name on picker cards
    /// and the detail view. Mirrors the mockup's `summary` field.
    /// Should evoke the character of the trail, not duplicate the stats
    /// (no distance / time / difficulty — those have their own row).
    /// Examples: "A loop through hemlocks older than the country."
    let summary: String
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
    /// Polyline of (lat, lng) points tracing the trail on the map.
    /// Drawn by TrailMapView as a lime overlay on top of Apple Maps.
    ///
    /// For Old Field the polyline is OSM-sourced — stitched from the
    /// real Old Field Trail [Red] + Jennings Trail [Blue] OSM ways
    /// via Overpass API. Every stop coordinate is a vertex on this
    /// polyline by construction. For Kildoo and Tranquil the
    /// coordinates are visual estimates matching the mockup's
    /// hand-placed trail (see design/README.md "Known limitations").
    let path: [CLLocationCoordinate2D]
    /// Post-tour "Discoveries" — the recap stream shown on the
    /// journal screen. Curator-authored per trail, ~5 cards each,
    /// each anchored by a hero number/date/quantity. See the
    /// design/mockups.html `.discoveries` block.
    let learnings: [Learning]
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
        summary: "A loop through hemlocks older than the country.",
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
                coordinate: c(40.9528830, -80.1700386),
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
                coordinate: c(40.9498610, -80.1715602),
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
                coordinate: c(40.9465772, -80.1733358),
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
                coordinate: c(40.9429821, -80.1750093),
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
                coordinate: c(40.9403631, -80.1760979),
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
        """,
        // Kildoo polyline. 79 vertices from design/mockups.html
        // TRAILS.kildoo.path. Visual estimate (Kildoo coordinates are
        // not OSM-sourced — see Trail.path doc).
        path: [
            c(40.9528830, -80.1700386), c(40.9524726, -80.1702225), c(40.9520972, -80.1705409),
            c(40.9519286, -80.1706393), c(40.9517501, -80.1707987), c(40.9514500, -80.1709961),
            c(40.9507056, -80.1712836), c(40.9503944, -80.1713490), c(40.9501791, -80.1713475),
            c(40.9501245, -80.1713633), c(40.9498610, -80.1715602), c(40.9497571, -80.1716061),
            c(40.9496678, -80.1716247), c(40.9495727, -80.1716975), c(40.9494733, -80.1717691),
            c(40.9493635, -80.1719394), c(40.9492600, -80.1720198), c(40.9492019, -80.1721223),
            c(40.9491388, -80.1721869), c(40.9490004, -80.1723537), c(40.9488547, -80.1725217),
            c(40.9487005, -80.1726453), c(40.9485759, -80.1726847), c(40.9484663, -80.1726778),
            c(40.9483250, -80.1726855), c(40.9482341, -80.1727043), c(40.9479536, -80.1727726),
            c(40.9477763, -80.1728682), c(40.9476256, -80.1729145), c(40.9475071, -80.1730353),
            c(40.9473174, -80.1731258), c(40.9471888, -80.1731415), c(40.9470092, -80.1732296),
            c(40.9469180, -80.1733093), c(40.9468352, -80.1733491), c(40.9467678, -80.1733105),
            c(40.9466520, -80.1733117), c(40.9465772, -80.1733358), c(40.9464012, -80.1734095),
            c(40.9463100, -80.1734216), c(40.9462141, -80.1734634), c(40.9460426, -80.1735018),
            c(40.9459794, -80.1735417), c(40.9458577, -80.1735773), c(40.9457689, -80.1735570),
            c(40.9456444, -80.1735999), c(40.9455541, -80.1736138), c(40.9454903, -80.1736341),
            c(40.9453691, -80.1737314), c(40.9452952, -80.1737375), c(40.9449824, -80.1739074),
            c(40.9448961, -80.1739330), c(40.9448492, -80.1739792), c(40.9447777, -80.1740219),
            c(40.9447156, -80.1740816), c(40.9445913, -80.1741489), c(40.9444746, -80.1741971),
            c(40.9443003, -80.1742991), c(40.9440836, -80.1743691), c(40.9436875, -80.1746211),
            c(40.9435443, -80.1746826), c(40.9434115, -80.1747013), c(40.9433161, -80.1748430),
            c(40.9429821, -80.1750093), c(40.9429034, -80.1751491), c(40.9427880, -80.1753258),
            c(40.9427325, -80.1753890), c(40.9424734, -80.1755406), c(40.9423125, -80.1755749),
            c(40.9420405, -80.1756967), c(40.9419799, -80.1757479), c(40.9417151, -80.1758867),
            c(40.9415080, -80.1759416), c(40.9414119, -80.1759407), c(40.9412146, -80.1759919),
            c(40.9411411, -80.1760534), c(40.9406894, -80.1759797), c(40.9405775, -80.1760499),
            c(40.9405369, -80.1760522), c(40.9404755, -80.1760524), c(40.9404088, -80.1760730),
            c(40.9403631, -80.1760979)
        ],
        // Recap "Discoveries" — verbatim from design/mockups.html.
        learnings: [
            Learning(
                anchor: "320 million years",
                body: "Age of the sandstone in the layered cliffs. The orange streaks are iron oxide leached out of the rock by groundwater over geologic time."
            ),
            Learning(
                anchor: "Iron oxide",
                body: "What turns the cliff face orange — leached out of the sandstone by groundwater, stain by stain, over a very long time."
            ),
            Learning(
                anchor: "Three centuries",
                body: "Age of the eastern hemlocks leaning over the gorge above Kildoo Falls — older than the country itself."
            ),
            Learning(
                anchor: "1874",
                body: "The Covered Bridge was built this year — Howe truss design, one of two left in Pennsylvania. The mill ground grain here until 1928."
            ),
            Learning(
                anchor: "80 tons",
                body: "Weight of Slippery Rock — the sandstone boulder in the creek that gave the waterway its name. Algae keeps it slick."
            )
        ]
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
        summary: "A wildflower loop through reclaimed farms.",
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
                coordinate: c(40.5085885, -80.3645201),
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
                coordinate: c(40.5091497, -80.3612003),
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
                coordinate: c(40.5086195, -80.3552001),
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
                coordinate: c(40.5009533, -80.3614658),
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
                coordinate: c(40.5068290, -80.3636062),
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
        """,
        // Old Field & Jennings polyline. 180 vertices, GPX-accurate.
        // Stitched from two real OSM ways via Overpass API: Old Field
        // Trail [Red] (W→E) + Jennings Trail [Blue] reversed (E→W),
        // joined at the east-end junction and closed back to the
        // trailhead in the southwest. Every stop coordinate is a
        // vertex on this polyline by construction.
        path: [
            c(40.5085885, -80.3645201), c(40.5088193, -80.3644860), c(40.5090111, -80.3645557),
            c(40.5090967, -80.3646523), c(40.5090784, -80.3648105), c(40.5091212, -80.3648347),
            c(40.5092089, -80.3647810), c(40.5093088, -80.3647542), c(40.5094883, -80.3648400),
            c(40.5096901, -80.3649071), c(40.5097656, -80.3648374), c(40.5098778, -80.3645772),
            c(40.5098655, -80.3643304), c(40.5098044, -80.3640622), c(40.5096983, -80.3638449),
            c(40.5096330, -80.3637671), c(40.5094862, -80.3636840), c(40.5093843, -80.3635499),
            c(40.5092680, -80.3634426), c(40.5091763, -80.3633219), c(40.5091538, -80.3631530),
            c(40.5091192, -80.3629008), c(40.5090682, -80.3626192), c(40.5090295, -80.3623348),
            c(40.5089886, -80.3621578), c(40.5090682, -80.3620130), c(40.5090845, -80.3617153),
            c(40.5091130, -80.3614283), c(40.5091497, -80.3612003), c(40.5093068, -80.3611118),
            c(40.5093741, -80.3610259), c(40.5094209, -80.3608703), c(40.5095005, -80.3607202),
            c(40.5095617, -80.3605351), c(40.5095820, -80.3603205), c(40.5095596, -80.3600952),
            c(40.5095148, -80.3599906), c(40.5093883, -80.3598914), c(40.5093724, -80.3598082),
            c(40.5093598, -80.3596312), c(40.5094781, -80.3593951), c(40.5094556, -80.3591725),
            c(40.5094495, -80.3590170), c(40.5095250, -80.3587836), c(40.5095413, -80.3585851),
            c(40.5094577, -80.3584027), c(40.5094597, -80.3582472), c(40.5096269, -80.3580809),
            c(40.5097207, -80.3579334), c(40.5098372, -80.3577388), c(40.5098574, -80.3572601),
            c(40.5095739, -80.3568444), c(40.5093129, -80.3562543), c(40.5089988, -80.3559485),
            c(40.5089152, -80.3557702), c(40.5088734, -80.3557487), c(40.5088156, -80.3557757),
            c(40.5087867, -80.3554764), c(40.5087867, -80.3553370), c(40.5087144, -80.3552324),
            c(40.5086195, -80.3552001), c(40.5084859, -80.3552552), c(40.5083911, -80.3553021),
            c(40.5082698, -80.3553987), c(40.5081505, -80.3554899), c(40.5079924, -80.3558131),
            c(40.5079414, -80.3560384), c(40.5079007, -80.3562368), c(40.5078558, -80.3565923),
            c(40.5077910, -80.3570347), c(40.5078079, -80.3573889), c(40.5078435, -80.3579789),
            c(40.5078772, -80.3582646), c(40.5078986, -80.3586441), c(40.5078476, -80.3591296),
            c(40.5077701, -80.3597908), c(40.5076916, -80.3601650), c(40.5075632, -80.3603393),
            c(40.5075672, -80.3605753), c(40.5074164, -80.3608811), c(40.5072634, -80.3612566),
            c(40.5071308, -80.3617179), c(40.5070166, -80.3619272), c(40.5067781, -80.3621525),
            c(40.5065965, -80.3623188), c(40.5064088, -80.3625306), c(40.5061723, -80.3625763),
            c(40.5060418, -80.3626809), c(40.5058338, -80.3628069), c(40.5056054, -80.3627425),
            c(40.5051302, -80.3628069), c(40.5048834, -80.3629357), c(40.5047141, -80.3630350),
            c(40.5044470, -80.3629893), c(40.5041145, -80.3629115), c(40.5037413, -80.3628659),
            c(40.5034966, -80.3626459), c(40.5032825, -80.3626031), c(40.5031988, -80.3625602),
            c(40.5031172, -80.3625011), c(40.5030266, -80.3624375), c(40.5028562, -80.3624234),
            c(40.5026706, -80.3622839), c(40.5025176, -80.3621042), c(40.5023096, -80.3619245),
            c(40.5020955, -80.3617018), c(40.5019813, -80.3616160), c(40.5019099, -80.3614685),
            c(40.5017732, -80.3611359), c(40.5016835, -80.3608730), c(40.5016019, -80.3607202),
            c(40.5014755, -80.3605780), c(40.5012633, -80.3605512), c(40.5011675, -80.3605619),
            c(40.5010736, -80.3606129), c(40.5010227, -80.3607067), c(40.5009982, -80.3608382),
            c(40.5009676, -80.3609750), c(40.5009595, -80.3611761), c(40.5009533, -80.3614658),
            c(40.5009798, -80.3617233), c(40.5009595, -80.3619808), c(40.5009941, -80.3622007),
            c(40.5009962, -80.3623483), c(40.5009595, -80.3624502), c(40.5009880, -80.3625494),
            c(40.5010512, -80.3626353), c(40.5011185, -80.3626889), c(40.5010868, -80.3629763),
            c(40.5010920, -80.3631529), c(40.5011083, -80.3634104), c(40.5011767, -80.3634386),
            c(40.5013918, -80.3634024), c(40.5016080, -80.3634077), c(40.5019058, -80.3632522),
            c(40.5020302, -80.3632817), c(40.5021750, -80.3633970), c(40.5023728, -80.3636357),
            c(40.5025074, -80.3638557), c(40.5027012, -80.3639174), c(40.5028440, -80.3638449),
            c(40.5029500, -80.3638449), c(40.5030663, -80.3639683), c(40.5031662, -80.3639201),
            c(40.5032763, -80.3637993), c(40.5034293, -80.3637698), c(40.5035129, -80.3637993),
            c(40.5036638, -80.3639120), c(40.5037780, -80.3639549), c(40.5039330, -80.3639174),
            c(40.5039963, -80.3639764), c(40.5040840, -80.3640380), c(40.5041553, -80.3640273),
            c(40.5042186, -80.3640005), c(40.5043226, -80.3639764), c(40.5043939, -80.3639361),
            c(40.5045653, -80.3638423), c(40.5047183, -80.3638100), c(40.5048141, -80.3637698),
            c(40.5048467, -80.3640300), c(40.5048549, -80.3642553), c(40.5049119, -80.3643680),
            c(40.5050017, -80.3644216), c(40.5050996, -80.3643841), c(40.5051730, -80.3643036),
            c(40.5053871, -80.3640837), c(40.5055299, -80.3639897), c(40.5057012, -80.3639442),
            c(40.5057726, -80.3639281), c(40.5058603, -80.3638557), c(40.5058786, -80.3637242),
            c(40.5060357, -80.3635418), c(40.5063049, -80.3633031), c(40.5064109, -80.3632790),
            c(40.5064803, -80.3633434), c(40.5065639, -80.3634506), c(40.5066125, -80.3634819),
            c(40.5068290, -80.3636062), c(40.5070437, -80.3638191), c(40.5085885, -80.3645201)
        ],
        // Recap "Discoveries" — authored for Old Field & Jennings.
        // Anchors lean toward the wildflower/reclaimed-farm themes
        // that distinguish this trail from the McConnells Mill gorge.
        learnings: [
            Learning(
                anchor: "Seven years",
                body: "How long large-flowered trillium takes to bloom from seed. The white-petaled carpet under the wildflower meadow is the result of decades of slow accumulation."
            ),
            Learning(
                anchor: "1800s farms",
                body: "The weathered fence posts in the understory mark the edges of farms that were here before the forest. The land reclaimed itself within a single human lifetime."
            ),
            Learning(
                anchor: "Spring ephemerals",
                body: "Trillium, Virginia bluebells, and Dutchman's breeches all bloom and seed in the brief window before the canopy closes. Most of the year they're invisible."
            ),
            Learning(
                anchor: "Pileated woodpecker",
                body: "The largest woodpecker in the eastern forest. The clean rectangular holes in dead snags here are its work — chiseled out chasing carpenter ants."
            ),
            Learning(
                anchor: "Raccoon Creek",
                body: "The slope at the east overlook drops toward this creek — the watershed that defines the park, and a tributary of the Ohio River."
            )
        ]
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
        summary: "A wooded creek walk in the heart of Pittsburgh.",
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
                coordinate: c(40.4460114, -79.9030646),
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
                coordinate: c(40.4391538, -79.9001871),
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
                coordinate: c(40.4296463, -79.9008283),
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
        """,
        // Tranquil polyline. 20 vertices traced from OSM way
        // "Tranquil Trail" in Frick Park. The walk is out-and-back,
        // so this path covers the one-way descent from Trailhead to
        // Forest Grove; the user retraces it on the return.
        path: [
            c(40.4460114, -79.9030646), c(40.4459570, -79.9037673), c(40.4457599, -79.9039644),
            c(40.4453903, -79.9038104), c(40.4449530, -79.9023937), c(40.4444020, -79.9019957),
            c(40.4439121, -79.9020426), c(40.4429959, -79.9020431), c(40.4419657, -79.9017159),
            c(40.4406303, -79.9007271), c(40.4391538, -79.9001871), c(40.4377749, -79.8990330),
            c(40.4366230, -79.8989346), c(40.4347568, -79.8991703), c(40.4338944, -79.8994021),
            c(40.4330023, -79.9000810), c(40.4322765, -79.9004010), c(40.4313517, -79.9006325),
            c(40.4302675, -79.9007073), c(40.4296463, -79.9008283)
        ],
        // Recap "Discoveries" — authored for Tranquil. Themes lean
        // toward Pittsburgh history + the geology of Fern Hollow.
        learnings: [
            Learning(
                anchor: "644 acres",
                body: "Frick Park is Pittsburgh's largest historic park — built up from Helen Clay Frick's 1919 bequest and continuously expanded since."
            ),
            Learning(
                anchor: "150+ years",
                body: "Some of the oaks and tulip poplars in Forest Grove pre-date the city's industrial era. They survived because this slope was too steep to log."
            ),
            Learning(
                anchor: "Skunk cabbage",
                body: "One of the first plants to bloom each spring — can melt its own snow with metabolic heat, sometimes visible against the late frost."
            ),
            Learning(
                anchor: "Fern Hollow Creek",
                body: "Drains into Nine Mile Run downstream. The whole watershed sat under industrial slag for most of the 20th century before the 2002 restoration."
            ),
            Learning(
                anchor: "Pittsburgh Coal",
                body: "The coal-bearing strata beneath the park formed about 300 million years ago, when this region was a coastal swamp near the equator."
            )
        ]
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

# Trailogy

Interactive prototype of an audio-first nature companion app.
Single-file HTML mockup that walks through trail selection, in-tour narration with location-triggered stops, on-demand voice questions, photo-as-context capture, and a post-hike **recap** (knowledge digest of what the user learned).

## Local preview

```bash
python3 -m http.server 8000
open http://localhost:8000/mockups.html
```

No build step, no package manager. Open the file directly in any modern browser.
The prototype renders an iPhone-shaped frame in the center of the page; navigate via the in-frame UI.

## File layout

```
mockups.html   – entire prototype (HTML + CSS + vanilla JS)
.gitignore     – macOS metadata + local-only video assets
CLAUDE.md      – this file
```

## Design philosophy

Audio-first. The phone stays in the pocket roughly 95% of the hike. The screen is a soft door into the place, not where the user lives.

- Photography over icons, type over chrome, stillness over motion.
- Dark theme; one accent — lime `#d9f571` — used for the active waypoint, primary CTAs, mic pulse, and the offline-ready dot. Nothing else.
- Inter sans-serif throughout, tight tracking on display sizes for outdoor legibility.
- No notifications, no streaks, no social, no splash screens. The discipline of those absences is the design.

## Architecture

### Views

Four named views, swapped via `go(viewName)` with opacity cross-fades:

| `data-view` | Role |
|---|---|
| `picker` | Landing: lime-glow heading *"Where should Trailogy take you?"* + lime location-pin + three trail cards with summary taglines (no profile / account chrome — anonymous by design) |
| `detail` | Full-screen Leaflet map + state-aware bottom CTA (Download → Begin) |
| `walking` | Tour-in-progress; state machine; three controls at the bottom (camera, mic, more) |
| `journal` | Post-hike recap (user-facing: **"Recap"**, internal name still `journal`). Trailmark + trail name + meta + Takeaways list. Knowledge digest, not trip report. |

### Tour state machine (walking view)

Located under `// ---------- tour journey state machine ----------`. Cycles automatically:

```
at-stop  ───────► between ───────► approaching ───────► at-stop (next idx)
9000 ms           5000 ms            3000 ms             …
```

After the final stop's `at-stop` window, the state becomes `complete` (no further transitions). The pause menu (`More` button → sheet) suspends the cycle; the end-tour item routes to `journal`.

Per-state UI:

- **at-stop** — hero card with stop image fades in *at the top* (replaces the progress bar); lyric narration centered vertically in the remaining space; progress-bar active marker pulses lime. If the stop has a `payoff` (callback to the previous stop's `lookFor` prompt), it's prepended to the narration so it's the first sentence the user hears on arrival.
- **between** — progress bar visible with a "you-are-here" pin sliding between markers; center shows *Walking to ▸ next stop name* with a 3-dot step animation and distance/time; below that, an *ON THE WAY* eyebrow + the previous stop's `lookFor` prompt invites the user to notice something specific on the walk.
- **approaching** — same layout as between but eyebrow + name go lime; next stop's marker pulses lime; distance line fades out. The look-for prompt stays visible.
- **complete** — lime-bordered checkmark, *TOUR COMPLETE / You walked the loop / 2.0 mi · 5 stops · 1 hr 12 min* + lime *Open journal* button.

Each cycle transition also fires a haptic via `navigator.vibrate` (Android Chrome / most mobile browsers honor it; iOS Safari ignores it — a production native iOS app would use `UIImpactFeedbackGenerator`). No audio cues — the original prototype had Web Audio chimes and then real-flute samples; both were stripped in favour of haptic-only for simplicity and to match the "phone in pocket" philosophy.

### Engagement: look-for / payoff prompts

Each stop (except the last) has a `lookFor` string and each stop (except the first) has a `payoff` string. The pair forms an arc:

1. At stop N, the at-stop narration plays normally.
2. As the user departs, the between/approaching panel surfaces *ON THE WAY · "Look for the mill race…"* — a single sentence inviting them to notice something specific on the walk.
3. On arrival at stop N+1, the first line of narration is the `payoff` — a one-line acknowledgement of what they were watching for ("If you saw a long stone-lined trench, that was the old mill race…").

The data lives on each stop:

```js
{ num, name, lat, lng, img,
  sentences: [...],
  payoff: "…",     // optional; callback for the previous stop's lookFor
  lookFor: "…" }   // optional; what to notice on the walk to the next stop
```

`applyStop()` prepends `payoff` to the lyric sentences when present; `enterBetween()` writes `lookFor` into `#wq-lookfor`. The `:empty` CSS rule auto-hides the element when no prompt exists (first / last stops of a trail).

### Trail data

Single `TRAILS` object at the top of the script. Keyed by `kildoo`, `oldfield`, `tranquil`. Each:

```js
{
  name, location, summary,        // summary = one-line tagline for picker + detail
  distance, distanceUnit,
  timeNum, timeUnit, difficulty,
  downloadSize, downloaded,       // for state-aware Download → Begin CTA
  walked,                         // false until the user finishes this trail in-session
  walkedDate,                     // null until completion; stamped to today (e.g. "May 16")
  stops: [
    { num, name, lat, lng, img,
      sentences: [...],
      payoff,    // optional · resolves prev stop's lookFor (omit on stop 1)
      lookFor }  // optional · prompts the walk to the next stop (omit on last)
  ],
  learnings: [                    // takeaway cards rendered in the recap
    { headline, flavor, category }  // category drives the corner icon
  ],
  path: [[lat, lng], …],          // every stop is a vertex on the polyline
  segmentDistances: [...]          // per-leg copy for the Walking-to indicator
}
```

`selectedTrail` (`'kildoo'` by default) and a `syncTrailRuntime()` helper keep `stopData`, `STOP_POS`, `SEGMENT_DISTANCES`, and the progress-bar marker DOM in sync whenever the active trail changes.

### Recap (the post-hike screen)

User-facing label: **"Recap"** (internal route name is still `journal`). Reached three ways:

1. Tapping the **Completed {date}** badge on a picker card. The badge only appears on trails the user has finished this session — it's injected at runtime by `renderPickerBadges()` based on `TRAILS[id].walked`, with `walkedDate` formatted from `new Date()` at completion time.
2. The "Open recap" button on the tour-complete screen
3. The "End tour" item in the More sheet

Layout, top to bottom, all centered:

- **Trailmark** — three filled lime rectangles in a triangle. Literally the U.S. trail-terminus blaze convention; means "trail ends here." Distinctive, brand-specific (`Trailogy → trail-mark`), works on its own without an "Complete" label.
- **Trail name** — `TRAILS[selectedTrail].name`
- **Meta line** — `{walkedDate} · {distance} · {stops} stops`, uppercase tracked, lime middots between items
- **"Takeaways" header** — small lime uppercase tracked label
- **Takeaway cards** — rendered dynamically by `renderRecap()` from `TRAILS[selectedTrail].learnings`. Each card: small lime category illustration (top-right) + 19 px cream headline sentence + 14 px dim cream flavor sentence. Soft staggered entrance animation.

`renderRecap()` updates the trail name + meta line + cards container from data; called from `go('journal')`. Tapping the picker journal-link badge sets `selectedTrail` first so the recap reflects that trail.

### Category icon system

Each takeaway carries a `category`; the illustration is looked up from `CATEGORY_ICONS` at render time by `paintCategoryIcons()`. Nine categories cover ~95% of trail content:

| Category | Illustration | Example content |
|---|---|---|
| `geology` | sedimentary strata + pebbles | rock age, cliffs, boulders |
| `water` | droplet | creeks, falls, hydrology |
| `plant` | leaf with vein | trees, wildflowers, ferns |
| `wildlife` | two birds in flight | animals, birds, insects |
| `history` | folded-corner page + date lines | events, settlers, place names |
| `architecture` | covered bridge | bridges, mills, built features |
| `sky` | sun with eight rays | weather, light, seasons |
| `chemistry` | three bonded atoms (triangle) | compounds, reactions |
| `other` | six-line asterisk | fallback for anything that doesn't classify |

For production with LLM-generated takeaways from user Q&A: the prompt is *"Summarize this exchange as `{ headline, flavor }`. Tag with one of: geology, water, plant, wildlife, history, architecture, sky, chemistry, other."* Reliable LLM classification task. No per-takeaway artwork needed.

### Map

Leaflet 1.9.4 via unpkg CDN with CARTO Dark Matter tiles. Trail rendered as a lime polyline with a soft drop-shadow. Stop markers are custom `divIcon` waypoints — dark glass for inactive, lime-filled with a breathing halo for active. Labels sit to the right of each pin with a 4-layer dark text shadow to stay readable across any tile.

Two map surfaces share this renderer so they always look identical:

- **Detail view** (`#detail-leaflet`, inside `.dm-canvas`) — full-screen trail map with the Begin CTA. `initDetailMap` initializes once; `rebuildDetailMap` swaps the polyline / markers when the selected trail changes. Active marker is always stop 1.
- **In-tour map** (`#tour-leaflet`, inside `.tm-canvas`) — full-screen overlay that slides up when the user taps the progress bar or the stop hero during a tour. `initTourMap(activeIdx)` / `rebuildTourMap(activeIdx)` mirror the detail pair but accept the tour state machine's `currentStopIdx` so the active marker pulses lime at the stop the user is actually at. Header rebinds to `<trail>` + `Stop N of M · <stop name>` on each open.

`.dm-canvas` and `.tm-canvas` share the same Leaflet styling selectors so basemap brightness, attribution chrome, and polyline drop-shadow are identical between them.

### Walking-view controls

Three buttons at the bottom of the walking view, all at uniform 28 px gap:

| Button | Size | Style | Purpose |
|---|---|---|---|
| 📷 `cam-btn` | 60 px | Lime outline | Optional photo context — opens viewfinder, photo becomes context for a follow-up voice question |
| 🎤 `ask-btn` | **84 px** | **Lime fill** (primary) | Press-and-hold voice question. Headline interaction. Lime ripple animation while held. |
| ⋯ `more-btn` | 56 px | Neutral outline | Utility menu (pause / end tour) |

Visual hierarchy: **size + color + fill** carry the meaning. Mic is biggest + filled (primary). Camera is medium + lime-outlined (related secondary — same color family, smaller size). More is smallest + neutral-outlined (separate utility).

**Camera tip popup** — on the user's first camera tap of a session, an iOS-style alert shows: *"I'm best at plants right now. Try a leaf, flower, or fern. I'll tell you what I see."* with **Cancel** and **Open camera** buttons. Tapping Cancel dismisses without marking the tip seen, so the user gets the explanation again next time. Tapping Open camera marks it seen and opens the viewfinder. Reset clears the flag.

In production iOS the same pattern uses `UserDefaults` for the seen flag and launches the system `UIImagePickerController` after dismissal — no custom AVFoundation camera needed.

### Photography

All trail-card and stop-hero images load from Wikimedia Commons direct URLs. No API keys, no third-party image services, public-domain or CC-licensed sources.

## Trails

| Trail | Park | Length | Difficulty | Duration | Stops |
|---|---|---|---|---|---|
| Kildoo Trail | McConnells Mill State Park | 2.0 mi loop | Moderate | ~1 hr | 5 |
| Old Field & Jennings Trail Loop | Wildflower Reserve, Raccoon Creek State Park | 2.3 mi loop | Easy | ~50 min | 5 |
| Tranquil Trail | Frick Park, Pittsburgh | 1.1 mi out-and-back | Easy | ~30 min | 3 |

Stats sourced from AllTrails / PA DCNR. Kildoo and Tranquil coordinates are geographic estimates; the Old Field & Jennings loop is stitched from real OSM ways (Old Field Trail [Red] + Jennings Trail [Blue]) via the Overpass API and closes back to the trailhead. Stops are exact vertices on the polyline by construction. Drop real GPX coordinates into `TRAILS[id].path` and `TRAILS[id].stops` to upgrade the other two.

## Major design decisions (history)

1. **Serif → Inter sans-serif.** First iteration leaned literary (Cormorant Garamond / Source Serif). Switched for outdoor sunlight legibility while keeping the restrained color palette.
2. **Five static screens → single interactive prototype.** Consolidated into one iPhone frame with `go()` view transitions.
3. **SVG illustrations → Leaflet + CARTO Dark Matter tiles.** Real cartography (rivers, roads, terrain labels) replaced the abstract trail diagram.
4. **Walking screen state machine.** Auto-cycles `at-stop → between → approaching → next`. Image hero swaps in *at the top* (replaces progress bar) when at a stop; quiet *walking to X* indicator with step animation in between.
5. **Three trails, not one.** Hardcoded Kildoo data extracted into a `TRAILS` object; picker cards each drive their own detail map, narration, and tour cycle.
6. **One overflow button for Pause + End.** Single `•••`-shaped button (renders as pause / play depending on state) opens a small sheet with both options.
7. **Tour completion state.** Instead of looping back to stop 1, after stop N the screen shows a completion summary with a journal link.
8. **Picker stats matched to AllTrails.** Replaced the original drive-time-from-Pittsburgh metric with actual hike duration to remove the mismatch with the detail view.
9. **Hells Hollow → Wildflower Reserve.** Swapped the third trail card for the Old Field & Jennings Loop at Wildflower Reserve (Raccoon Creek State Park). Geometry is stitched from real OSM ways into a closed 2.3 mi loop with all 5 stops landing exactly on the polyline.
10. **Unified in-tour map.** The map overlay that opens from the progress bar / stop hero used to be a hardcoded SVG of the Kildoo loop; replaced with a second Leaflet instance that shares the detail view's polyline, markers, and styling, parameterised by the live `currentStopIdx` so the active waypoint pulses lime at the right stop on every trail.
11. **iOS-style Begin alert.** When the user taps Begin off-site, a UIAlertController-look modal (`.ios-alert`) explains the tour will play in time sequence instead of GPS-triggered. Replaces a plainer in-screen prompt and keeps the iOS framing consistent.
12. **Audio chimes explored and removed.** Tried Web Audio bell synthesis, then wind/thump textures, then real flute samples for between-stops cues. Each iteration was wrong in its own way (notification-y / synthetic / asset overhead). Settled on **haptic only** — matches the "phone in pocket" philosophy and the discipline-of-absences design ethos. Real iOS shipping would use `UIImpactFeedbackGenerator`; the web prototype uses `navigator.vibrate`.
13. **Look-for / payoff engagement loop.** Each stop carries an optional `lookFor` (invitation to notice something on the walk to the next stop) and `payoff` (callback resolution on arrival). The between-stops space stops being dead time and becomes a small game of attention — *museum audio guide*, not a podcast. Pure data + ~30 lines of JS / CSS; no gamification, no streaks, no points.
14. **Profile button removed.** No account system in v1, so the placeholder profile chip in the picker's top-right was deleted. The picker corners stay empty by design.
15. **Demo-mode framing for hackathon judges.** The production app is location-based — each stop unlocks via Core Location when the user arrives. The prototype can't be at the trail, so the iOS-style alert on Begin carries the entire framing: *"Tours are location-based · On the trail, stops play when you arrive. This demo will auto-advance."* — primary button is *Begin Tour*. (An earlier iteration also kept a persistent `.demo-badge` on the walking view, but the alert proved clear enough on its own; the badge was removed to keep the walking screen free of demo-only chrome.) To ship, replace the AT_STOP_MS / BETWEEN_MS / APPROACHING_MS timers with Core Location region monitoring; the alert becomes a true error path that only fires when GPS is denied or the user is far from the trailhead.
16. **Journal reframed as "what you learned."** First iteration was a trip report: route map + five photo-cards of the stops + sightings + share-when-connected. Replaced with a knowledge digest — four curator-authored **learning cards**, each anchored by a hero number/date/quantity (320 million years, 1874, 80 tons…) and a one-paragraph context. A separate **You asked** section carries the user's Q&A as the personal layer, visually distinct (italic question, lime left-border, soft lime tint). The route map became a full-bleed hero at the top with the **Loop closed** achievement card hanging over it; the lime check seal sits as a medallion at the boundary like a wax stamp. Stats reframed as a 3-column row (`2.0 · MILES · 5 · STOPS · 1:12 · HOURS`). Removed: per-stop photo cards, "What you saw" sightings, the share-when-connected button, the closing quote, the "Until next time" sign-off. The journal is now a takeaway, not a receipt.
17. **Download moved from picker cards to the detail-view CTA.** First iteration put a `Download · 68 MB` pill on each picker card; the user had to make a file-management decision *before* seeing the trail. New flow puts the download as a state-aware CTA on the detail view — same lime button cycles through `download` (label "Download · 68 MB" + arrow icon) → `downloading` (animated dark progress fill + percentage) → `ready` ("Begin" + play icon). Single button position, three states. Picker cards become pure choice: photo, region, name, stats, plus a journal-link badge for completed walks. Trail data carries `downloadSize` + `downloaded: bool`. Old Field starts `downloaded: true` (already walked once); Kildoo and Tranquil start `downloaded: false`.
18. **Picker heading + lime glow.** Opening copy *"Where should Trailogy take you?"* (17 px regular, with a small lime location-pin), centered above a soft lime radial gradient anchored top-left of the picker. Replaces an earlier "Start exploration · Trails nearby" pair. Warmer voice; the gradient is the only chrome decision visible on the landing screen.
19. **Per-trail summary tagline.** Each trail card now carries a one-line `summary` (e.g., *"A loop through hemlocks older than the country."*) displayed between the trail name and the stats. Same string surfaces on the detail view's bottom action card. Differentiates trails by feel/character, not just by distance and difficulty.
20. **Recap renamed user-facing: "Journal" → "Recap".** Internal route name stays `journal` for code stability; user-facing label is "Recap" everywhere it appears (Open recap button, badge aria-label).
21. **Recap redesign #2 — trailmark + 9-category dynamic content.** Several iterations on the recap converged on the current form. Removed: the route map at the top, the lime-bordered "Loop closed" achievement card with hanging seal medallion, the "You walked the trail" hero line, the closing quote and sign-off, the per-stop photo cards. Replaced with: a brand-specific **trailmark** at the top (three filled lime rectangles in a triangle — the U.S. trail-end blaze convention), followed by trail name + uppercase tracked meta line with lime middots, a centered "Takeaways" header in lime uppercase, and **dynamic per-trail learning cards** rendered from `TRAILS[id].learnings`. Each card is `{ headline, flavor, category }`. The category drives a small lime corner illustration drawn from a 9-icon `CATEGORY_ICONS` lookup (geology, water, plant, wildlife, history, architecture, sky, chemistry, other). All three trails have 5 authored learnings (15 total). `renderRecap()` is called from `go('journal')` and populates the header + cards from `TRAILS[selectedTrail]`. Same `{ headline, flavor, category }` shape works for future LLM-generated takeaways from user Q&A — no per-takeaway artwork needed.
22. **Walking-view controls hierarchy.** Three iterations: (a) original — mic primary, camera + more equal size. (b) Camera as twin primary with mic — rejected: photo is an addition to asking, not a coequal feature. (c) Final — mic biggest + lime fill (primary "ask"), camera medium + lime outline (related secondary — provides photo context), more smallest + neutral outline (utility menu). Equal gaps between all three. Workflow teaching happens via the camera-tip popup (first tap) and the photo-context strip (after capture), not via persistent button captions.
23. **Completed-badge: dynamic, stamped at finish.** First iteration hardcoded a "Completed Apr 14" badge into the Old Field picker card's HTML, with each trail's `walkedDate` set to a demo string. Replaced with runtime logic: all three trails start `walked: false, walkedDate: null` (clean slate). When the user finishes a tour — either naturally via `enterComplete()` or by tapping End Tour from the more menu — `markTrailWalked(selectedTrail)` stamps `walked = true` and `walkedDate = todayLabel()` (formatted from `new Date()` via `Intl`, e.g. `"May 16"`). `renderPickerBadges()` runs on every `go('picker')` and idempotently injects / removes the badge button based on each trail's `walked` state. Reset clears `walked` + `walkedDate` across all trails. The picker now reflects the user's actual session history rather than a fixed demo state; in production the same fields would be persisted to the device (localStorage on web, `UserDefaults` / Core Data on iOS) so completions survive relaunch.

## Design tokens

```
Background       #0a0c0a   app / picker
                 #050706   walking screen (deeper black)
Glass card       rgba(15,16,13,0.92)

Text             #f5f3ec   primary (soft off-white parchment)
                 #c8c5bc   secondary
                 #8a8881   meta
                 #5a5852   dim

Accent           #d9f571   lime — single accent. Used for: active waypoint, primary CTAs
                                    (Begin / Download / End tour / Open recap), mic + camera
                                    buttons, ON THE WAY eyebrow, trailmark in recap,
                                    location-pin in picker heading, soft picker glow,
                                    category icons in recap cards, lime middots in stats lines.
                 #c1dd58   lime pressed
```

Typeface: Inter 400 / 500 / 600 / 700 via Google Fonts. Display sizes use `-0.022em` letter-spacing.

## Vocabulary

Plainspoken, slightly literary. Used surfaces:

> Where should Trailogy take you? · Begin · Download · Hold to ask · Listening · Walking to · Approaching · ON THE WAY · Tour complete · You walked the trail · Loop closed · View recap · Completed {date} · Takeaways · Photo in context · Ask a follow-up question · I'm best at plants right now · Snap a plant to identify

## Known limitations

- **Coordinates are estimates for Kildoo and Tranquil** — visually plausible but not GPX-precise. The Old Field & Jennings loop is OSM-sourced and accurate. Real `.gpx` data would replace the `path` / `stops` arrays for the other two.
- **Photo context** uses a single cave image regardless of what was photographed. In production, the captured photo + a model-generated set of follow-up questions would replace `photoSentences` / `photoQuestions`. The image-recognition model (fine-tuned Gemma) is currently tuned for plants; the camera-tip popup names that limit upfront.
- **Trail distances:** AllTrails lists Kildoo as 3.1 mi out-and-back in one place; DCNR signage calls the loop 2.0 mi. We use 2.0 mi.
- **Completion history is in-memory only.** `walked` + `walkedDate` are stamped when the user finishes a tour but don't survive a page reload (same scope as the `downloaded` state). Production would mirror both into `localStorage` on web or `UserDefaults` / Core Data on iOS so the picker badges persist across launches.
- **Recap takeaways are static per trail.** Production would augment the base set with 1–2 dynamically-generated takeaways from the user's voice Q&A during the walk. The `{ headline, flavor, category }` data shape and `renderRecap()` already handle dynamic content; only the augmentation source needs wiring.
- **Tour completion summary on the walking screen** still reads `2.0 mi · 5 stops · 1 hr 12 min` regardless of which trail. Should be derived from `TRAILS[selectedTrail]` to match the picker / detail / recap metadata.

## Repo

[github.com/YingCeci/Trailogy-UI](https://github.com/YingCeci/Trailogy-UI) — private.

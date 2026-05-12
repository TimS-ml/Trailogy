# Trailogy

Interactive prototype of an audio-first nature companion app.
Single-file HTML mockup that walks through trail selection, in-tour narration with location-triggered stops, on-demand voice questions, photo-as-context capture, and a post-hike journal.

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
| `picker` | Landing: trail cards with download flow, profile button top-right |
| `detail` | Full-screen Leaflet map + Begin action |
| `walking` | Tour-in-progress; state machine |
| `journal` | At-home reading: per-stop entries, observations, share-when-connected |

### Tour state machine (walking view)

Located under `// ---------- tour journey state machine ----------`. Cycles automatically:

```
at-stop  ───────► between ───────► approaching ───────► at-stop (next idx)
9000 ms           5000 ms            3000 ms             …
```

After the final stop's `at-stop` window, the state becomes `complete` (no further transitions). The pause menu (`More` button → sheet) suspends the cycle; the end-tour item routes to `journal`.

Per-state UI:

- **at-stop** — hero card with stop image fades in *at the top* (replaces the progress bar); lyric narration centered vertically in the remaining space; progress-bar active marker pulses lime.
- **between** — progress bar visible with a "you-are-here" pin sliding between markers; center shows *Walking to ▸ next stop name* with a 3-dot step animation and distance/time.
- **approaching** — same layout as between but eyebrow + name go lime; next stop's marker pulses lime; distance line fades out.
- **complete** — lime-bordered checkmark, *TOUR COMPLETE / You walked the loop / 2.0 mi · 5 stops · 1 hr 12 min* + lime *Open journal* button.

### Trail data

Single `TRAILS` object at the top of the script. Keyed by `kildoo`, `oldfield`, `tranquil`. Each:

```js
{
  name, location, distance, distanceUnit,
  timeNum, timeUnit, difficulty,
  stops: [
    { num, name, lat, lng, img, sentences: [...] }
  ],
  path: [[lat, lng], …],          // every stop is a vertex on the polyline
  segmentDistances: [...]          // per-leg copy for the Walking-to indicator
}
```

`selectedTrail` (`'kildoo'` by default) and a `syncTrailRuntime()` helper keep `stopData`, `STOP_POS`, `SEGMENT_DISTANCES`, and the progress-bar marker DOM in sync whenever the active trail changes.

### Map

Leaflet 1.9.4 via unpkg CDN with CARTO Dark Matter tiles. Trail rendered as a lime polyline with a soft drop-shadow. Stop markers are custom `divIcon` waypoints — dark glass for inactive, lime-filled with a breathing halo for active. Labels sit to the right of each pin with a 4-layer dark text shadow to stay readable across any tile.

Two map surfaces share this renderer so they always look identical:

- **Detail view** (`#detail-leaflet`, inside `.dm-canvas`) — full-screen trail map with the Begin CTA. `initDetailMap` initializes once; `rebuildDetailMap` swaps the polyline / markers when the selected trail changes. Active marker is always stop 1.
- **In-tour map** (`#tour-leaflet`, inside `.tm-canvas`) — full-screen overlay that slides up when the user taps the progress bar or the stop hero during a tour. `initTourMap(activeIdx)` / `rebuildTourMap(activeIdx)` mirror the detail pair but accept the tour state machine's `currentStopIdx` so the active marker pulses lime at the stop the user is actually at. Header rebinds to `<trail>` + `Stop N of M · <stop name>` on each open.

`.dm-canvas` and `.tm-canvas` share the same Leaflet styling selectors so basemap brightness, attribution chrome, and polyline drop-shadow are identical between them.

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

## Design tokens

```
Background       #0a0c0a   app / picker
                 #050706   walking screen (deeper black)
Glass card       rgba(15,16,13,0.92)

Text             #f5f3ec   primary (soft off-white parchment)
                 #c8c5bc   secondary
                 #8a8881   meta
                 #5a5852   dim

Accent           #d9f571   lime (single accent; active waypoint, primary CTA, mic pulse)
                 #c1dd58   lime pressed
```

Typeface: Inter 400 / 500 / 600 / 700 via Google Fonts. Display sizes use `-0.022em` letter-spacing.

## Vocabulary

Plainspoken, slightly literary. Used surfaces:

> Begin · Ask · Download · Listening · Walking to · Approaching · Tour complete · You walked the loop · Send when connected · Share when connected · Press and hold to ask anything · Hold the screen to ask anything

## Known limitations

- **Coordinates are estimates for Kildoo and Tranquil** — visually plausible but not GPX-precise. The Old Field & Jennings loop is OSM-sourced and accurate. Real `.gpx` data would replace the `path` / `stops` arrays for the other two.
- **Journal content is hardcoded for Kildoo.** The *Completed Apr 14* link on the Old Field & Jennings card opens the same Kildoo journal page. A per-trail journal would mirror the per-trail tour data structure.
- **Photo context** uses a single cave image regardless of what was photographed. In production, the captured photo and a model-generated set of follow-up questions would replace `photoSentences` / `photoQuestions`.
- **Trail distances:** AllTrails lists Kildoo as 3.1 mi out-and-back in one place; DCNR signage calls the loop 2.0 mi. We use 2.0 mi.
- **Tour completion summary** still reads `2.0 mi · 5 stops · 1 hr 12 min` regardless of which trail the user just finished. The numbers should be derived from `TRAILS[selectedTrail]` to match the picker / detail metadata.

## Repo

[github.com/YingCeci/Trailogy-UI](https://github.com/YingCeci/Trailogy-UI) — private.

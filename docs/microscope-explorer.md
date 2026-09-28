# The microscope explorer: code, build, publish

The explorer is the interactive 3D site built from one recorded YuE2 generation. The recording
itself comes from `bin/microscope.py` (see [microscope.md](microscope.md)). This document covers
everything after that: the analysis that feeds the pages, the page code, the two builds, and
publishing to www.newty.coffee.

## What lives where

| What | Where |
|---|---|
| Page code (HTML, CSS, JS) | `src/audiogen/explore_assets/` in this repo |
| Data builders (Python) | `src/audiogen/explore.py` |
| Command line: build, serve, publish | `bin/explore.py` |
| Analysis scripts | `bin/find_entries.py`, `bin/machine_listen.py`, `bin/hear_words.py` |
| Tests | `tests/test_explore.py` |
| The recorded run (input) | `~/dev/projects/audiogen/output/microscope-runs/20260926-slow_down-s4417-take2/on-cpu/` |
| Local build (marks saved to disk) | `~/dev/projects/audiogen/output/microscope-runs/00_explore/` |
| Publish build (what goes on the web) | `~/dev/projects/audiogen/output/microscope-runs/00_publish/` |
| Pages on the web | repo `newty-coffee/www`, folder `yue2-microscope/` (GitHub Pages) |
| Audio on the web | Cloudflare R2 bucket `newty-media`, prefix `yue2-microscope/`, served at `media.newty.coffee` |

`RUNS` below means `~/dev/projects/audiogen/output/microscope-runs`, and `RUN` means the recorded
run folder. `publish` and `all` use `DEFAULT_RUN` (the Slow Down take 2 run) when no run is given; `focus` and
`stack` need one.

## The pages

Two pages are published, plus a landing page:

- **Coming into focus** (`focus.html`, `app/focus.js`): the whole song as a spectrogram landscape,
  redrawn at each of the 32 solver steps. The page has:
  - a dial, Play and Resolve;
  - the State, Predicted and Finished views, with Ghost, Settling color and Hz band toggles;
  - the "I hear…" listening game, with its pins, a time bar and an arrangement bar;
  - a panel of per-step numbers.
- **From plan to sound** (`stack.html`, `app/stack.js`; formerly "The stack"): the score, semantic tokens, latent and audio,
  stacked on one time axis, with a phrase list and a sped-up illustration of the stages.
- **Index** (`index.html`): cards linking to the two pages.

A third page, **Take map** (`map.html`, `app/map.js`), exists locally only. Publish removes it,
because without every take it says nothing.

### Shared code: `app/common.js`

Everything both pages use lives on `window.EX`:

- `stage`: the three.js renderer, camera, orbit controls and bloom, with a frame loop that pauses
  when the tab is hidden.
- `labels`: HTML labels pinned to 3D points and re-projected every frame.
- `nav`: the top bar.
- `help`: the "?" card, which opens the first time a page is visited. It moves the page's
  `.keys` list into itself.
- `tip`, `spark`, `dust`, `floor`, the color ramp (`rampJS`, `GLSL_RAMP`).
- **The glossary** (`GLOSSARY`, `g(key, text)`). Writing `<dfn data-g="latent">latent</dfn>`
  anywhere gives that word a dotted underline. Hovering it opens a card with a plain meaning
  and one technical line. To add a term, add an entry to `GLOSSARY`. `focus.js` adds the Hz
  band entries (`hz0`…`hz4`) at runtime.

three.js is vendored under `explore_assets/vendor/three/`; nothing loads from a CDN.

### Coming into focus: how `focus.js` is laid out

- **Landscape:** one mesh whose heights come from two data textures, state and predicted. Each
  texture is columns × bands × 33 steps, and the vertex shader moves between steps.
  - `W = 150` wide and `DEPTH = 78` deep, so the Hz bands spread out; `H = 17` tall.
  - Bass is at the front. Depth is log-frequency, from 40 Hz to 16 kHz.
- **Settling color:** each band's level envelope over the whole song is correlated with step 32's
  (`settle_state`, `settle_predicted`). Unsettled land is frost teal (`FROST` in the fragment
  shader) and blends toward the color that spot has in the finished song as its band settles. It is a
  comparison with the end result, per band and across the whole song, not a per-spot or per-instrument reading.
- **Ghost:** the other view, drawn as contour lines above the land.
- **Settling wall:** one bar per band at the left edge, showing the same correlation.
- **Hz bands:** `PITCH` holds five named bands (bass, body, voice & lead, bite, air), each with
  a color and plain and technical text.
  - `B` shows see-through strips above the land.
  - Hovering a band, on the land or at its label, fills its whole volume with a glowing block
    and colors the settling wall.
- **Time bar and arrangement bar:** in front of the land. Sections come from the score, so they
  are in score time.
- **Audio:** Web Audio. All layers are scheduled on one clock, so a step change is a crossfade.
  - The web build (`.m4a`) downloads every file before anything plays, behind the "Preparing the song"
    bar (bytes counted against `audio_bytes`, which publish writes). Then step 0 and the finished take
    are decoded and Play and Resolve turn on. Other files decode when their step is wanted, eight held
    at a time: decoding all 66 would hold about 2.3 GB.
  - The local build points at WAV listening copies (~35 MB each), loaded on demand with the steps
    either side decoded ahead.
  - Resolve takes its step from the song's clock (step × pace), so step 32 lands one second before the
    last chorus (`RESOLVE_BY`, from the sections) whichever step it starts from.
  - An "I hear" tap records the step that is sounding, and the buttons wait while a new step loads.
- **The listening game:** `CUES` come from `D.entries`, estimated times each part first comes in. The
  dots record where the listener tapped on that pass, not the earliest step a part can be heard.
  - A part's "I hear…" button appears once the song reaches that part, up to three at a time.
  - A tap stores `{step, at}` for that part, flies a dot to the time bar, puts a pin there, and
    puts a dot on the dial at the step.
  - Pins close together share a stacked tag, and each part gets its own leader line
    (`placePins`, run every frame).
  - Marks are stored in localStorage under `microscope-marks-<run>-cues`. Served locally, a tap
    also writes the matching listening field to the run on disk (`PARTS[id][2]`).
- **Hover help:** panel controls carry `data-help` (plain text, then a `<small class="tech">`
  line). The area under the note box shows the text for whatever the pointer is on.
- **Camera framing:** `frameView` fits the scene into the open space between the title, the
  panel and the dock, then shifts the projection (`setViewOffset`) so the land is centered there.

### From plan to sound: how `stack.js` is laid out

Four rows built from the score (ABC), the semantic tokens, the latent and the audio
spectrogram. Hovering a point lights that moment on every row. Clicking plays the phrase. The
replay is an illustration: token progress is interpolated between the run's recorded checkpoints, a
reused score is shown whole, and the acoustic stage is animated with no clock ("timing not recorded
here"). The lyrics and style prompt feed every stage; the page says so in its intro.

## Data: from the run to `data/*.js`

`src/audiogen/explore.py` reads the run and writes plain JS files that set a global:
`data/focus.js` sets `window.FOCUS`, `data/stack.js` sets `window.STACK`, and `data/summary.js`
feeds the index.

`focus_data(run)` reads:

| Field | From |
|---|---|
| state / predicted landscapes, settle maps | the per-step raw audio (`flow_audio_raw/chunk_000`, `flow_predicted_audio_raw/chunk_000`), as envelope grids |
| `metrics` | `flow/chunk_000/metrics.json`, plus `analysis/envelopes.json` and `analysis/bands.json` when present |
| `sections` | the plan's ABC score, in score time |
| `listening` | `analysis/annotations.json`: the author's own marks. Publish clears them |
| `machine` | `analysis/machine_marks.json`. Still built, but the focus page no longer shows it |
| `entries` | `analysis/entries.json`: when each part comes in |
| `audio` | paths to every step's peak-normalized listening copy (`flow_audio_listening`, …), relative to the build folder |

`focus_data` reads `chunk_000` only and refuses a run with more than one acoustic chunk.

Envelope grids are cached in `00_explore/cache/`, keyed by resolved path, file size, mtime and the
grid parameters. Publish and the default local build share it; a local build with its own `--out` uses
its own.

## The analysis scripts

Run these on a run folder before building. Each writes into `RUN/analysis/`.

| Script | Runs under | Writes | What it does |
|---|---|---|---|
| `bin/hear_words.py RUN` | this repo's `.venv` + audio.cpp | `heard.json` | Qwen3-ASR and the forced aligner on the finished take, section by section: the words heard and when |
| `bin/machine_listen.py RUN` | `~/dev/ai/sheetsage/.venv/bin/python` | `machine_marks.json`, `machine_listening/` | For each step, when a recognizer, a voice detector and SheetSage2 first pick out words, voice and chords |
| `bin/find_entries.py RUN [--by-ear id=seconds]` | this repo's `.venv`; separation shells out to `~/dev/ai/demucs/.venv` | `entries.json`, `stems/` | When each part first comes in, in song time |

How `find_entries.py` works:
- The results are estimates: stem activity is not proof that the named instrument is audible, and
  htdemucs_6s is known to bleed, piano especially.
- It separates the finished take with htdemucs_6s. Stems are used only to detect entries; nobody
  listens to them.
- A stem has entered once its level keeps coming back above −20 dB (relative to its own 95th
  percentile), with gaps no longer than 1.5 s, for at least 1 s. A stem more than 30 dB under
  the mix is treated as bleed and dropped.
- Words come from `heard.json` and chords from SheetSage.
- `--by-ear` records a listener's correction next to the machine's reading. For Slow Down it is
  run with `--by-ear drums=21.9`.

## The two builds

Run all commands from `~/dev/projects/audiogen-yue2`.

### Local: `00_explore` (for working on the pages)

```
.venv/bin/python bin/explore.py focus RUN      # or: stack RUN, map ROOT..., all
.venv/bin/python bin/explore.py serve          # http://127.0.0.1:8771/00_explore/index.html
```

- The pages refer to the run's audio by relative path.
- `serve` is required, because browsers refuse Web Audio fetches from `file://`. It also accepts
  `POST /api/annotate`, which is how marks made locally are written to
  `RUN/analysis/annotations.json`.
- Editing a file in `explore_assets/` does not change `00_explore` until you rebuild, or copy
  the file across (the build copies assets as they are).

### Publish: `00_publish` (what goes on the web)

```
.venv/bin/python bin/explore.py publish [RUN] --media-base <base>
```

`cmd_publish` does the following:

1. It replaces `00_publish` completely. It only deletes a folder that carries its own
   `.microscope-export` marker.
2. It copies the assets and drops the Take map.
3. It builds focus and stack data. Your own marks are stripped and the annotate endpoint is
   switched off, so a visitor's marks stay in their browser. The page is labeled with `--title` and
   `--take`, which are required for any run other than `DEFAULT_RUN` (labeled "Slow Down, take 2").
4. It converts every audio file the data refers to into `.m4a` twice, and rewrites each path to
   `<media-base>/<name>`:
   - **standard** (the default the pages load): Apple's AAC encoder (`aac_at`) at 64k, set by
     `--aac-encoder` and `--aac-bitrate`;
   - **HD**: `aac_at` at 160k, set by `--hd-encoder` and `--hd-bitrate`. The data carries `hd.map`
     (standard URL → HD URL). The "HD audio" button in the top bar stores the choice in localStorage
     (`microscope-hd`) and reloads, and both pages swap their URLs through the map.

   64k was chosen by ear against 160k on 2026-09-27 (State at early steps and the finished chorus):
   the difference was slight. `aac_at` is macOS only, so publish runs on the Mac.
    The name hashes the source bytes, the encoder settings (`aac_args`) and the
   ffmpeg version (`media/<sha256[:16]>.m4a`), so any change that could change the output gives a new
   name, and the files can be cached forever.
5. It rewrites the index for the web (`_swap`) with an intro, an About panel and two columns.
   `_swap` fails loudly if the text it expects has changed, so edits to `index.html` may need a
   matching edit in `cmd_publish`.
6. It copies the finished take's HD file to a readable name (`slow-down-take-2-<hash>.m4a`) for the
   pages' download button (`D.download`), and writes `media-manifest.json` (base, files, bytes, and
   `attachments`: files to upload with `--content-disposition 'attachment; filename="…"'`, so the
   browser saves them under that name).
7. **The leak scan:** the publish fails if any shipped `.html`, `.js`, `.css` or `.json`
   contains the home path, `/Users/`, `localhost`, `127.0.0.1`, `.internal` or `bin/explore.py`.
   Files under `vendor/three/` are skipped. It checks those fixed strings only. Keep them out of
   shipped assets, comments included.

Check a build with `--media-base media` (the default) and a plain static server:

```
cd RUNS/00_publish && python3 -m http.server 8000     # http://127.0.0.1:8000/
```

Publish deletes and recreates the folder. A server started from inside it, as above, keeps serving
the deleted folder, so restart it after every publish.

As of 2026-09-27, `media/` holds 132 files (66 standard, 66 HD), 174 MB (decimal). The focus page
downloads 49.5 MB before it plays (122 MB in HD).

## Publishing to www.newty.coffee

The pages and the audio go to different places. That is a choice, not a size limit (GitHub Pages
allows 1 GB): it keeps 125 MB of binaries out of the site repo and serves them from R2's cache.

1. **Build for the web:**
   `.venv/bin/python bin/explore.py publish --media-base https://media.newty.coffee/yue2-microscope`
2. **Upload the audio to R2** (run wrangler from outside `00_publish/`: it leaves a `.wrangler/` cache,
   with the account ID and name, in the folder it runs from): every file in `00_publish/media/`, standard and HD, goes to bucket `newty-media`,
   under `yue2-microscope/`, with a long cache lifetime (the names are content hashes). For
   example:
   `npx wrangler r2 object put newty-media/yue2-microscope/<file> --file <path> --remote --cache-control "public, max-age=31536000, immutable"`
   CORS on the bucket was set to allow `https://www.newty.coffee` (the apex domain redirects to
   www); check it again before the first upload.
3. **Pages to GitHub:** copy everything in `00_publish/` except `media/` and dot-folders into repo
   `newty-coffee/www` under `yue2-microscope/`, then commit and push to `main`. GitHub Pages
   serves it at `https://www.newty.coffee/yue2-microscope/`.
4. **Check the live site:** `EXPLORER_URL=https://www.newty.coffee/yue2-microscope/ npm run test:pages`
   runs the page suite against it in Chromium and WebKit: pages load without errors (Cloudflare's
   injected analytics beacon aside), audio plays, and nothing draws while nothing moves.

Live since 2026-09-27: https://www.newty.coffee/yue2-microscope/ (www commit 39b7741; the idle-drawing fix is b96b374, 2026-09-28), with its 132
audio files under `newty-media/yue2-microscope/`. The upload used one `wrangler r2 object put` per file
with `--content-type audio/mp4`, eight at a time. Files from earlier builds are never deleted
automatically; the names are content hashes, so new builds upload alongside them.

## Tests

```
.venv/bin/python -m pytest -q        # includes tests/test_explore.py
```

The Python tests cover the builders. The pages have a browser suite of their own:

```
npm install                           # once: Playwright 1.63, the version the Soundroom uses
npm run test:pages                    # rebuilds with `explore.py all`, serves with `serve --no-open`, drives Chromium
```

`tests/pages/explorer.spec.ts` counts WebGL draw calls (wrapped before any page script runs) to hold the
pages to drawing only while something moves: each page opens without errors, stops drawing when left alone,
wakes on input and goes quiet again, keeps drawing while its audio plays (including sound that starts after
its download), and the map's ambient turn runs only while its window has focus. It needs this machine's
microscope runs, since it builds from them.

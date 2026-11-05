# Delivery and demo reproduction

The prepared files are in the local `deliverables/` directory, which is deliberately
excluded from Git. Publish those files as release assets; do not commit footage,
credentials, local environments, or private signing keys.

| Artifact | Purpose |
| --- | --- |
| `mercymek-3-minute-demo.mp4` | Exactly 180 seconds, 1920 × 1080, 30 fps, H.264/AAC; English ElevenLabs narration and burned-in captions |
| `mercymek-3-minute-demo-v6.mp4` | Voice automation, visual geolocation, same-frame comparison, actual web-app 3D view and Evidence |
| `mercymek-demo.en.srt` | Separate English subtitles, timed from ElevenLabs character alignment |
| `mercymek-narration.mp3` | Edited, loudness-normalized narration |
| `mercymek-thumbnail.png` | 1080p cover image |
| `mercymek-source.zip` | Current application source, including uncommitted work; no credentials or data |
| `mercymek-demo-data.zip` | London case shown in the film, plus hero and snow examples and cached replies |
| `mercymek-extra-cases.zip` | Four additional cases: cut-in, lane change, night/rain and Las Vegas |
| `SHA256SUMS.txt` | Checksums for the deliverables |
| `release-validation.json` | Clean extraction, all original seals, range requests and commands with no API keys |
| `video-validation.json` | Encoding, duration and full video decode checks |
| `chapters.txt`, `narration.txt`, `CREDITS.txt` | Upload chapters, transcript, footage attribution and production credits |

## Judge quick start

Extract `mercymek-source.zip` into an empty directory. Install Python 3.11 or newer
and the project dependencies, then unpack both data archives from that directory:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux instead:
# source .venv/bin/activate
python -m pip install -e .
python scripts/pack_demo.py --unpack mercymek-demo-data.zip
python scripts/pack_demo.py --unpack mercymek-extra-cases.zip  # optional: four more cases
python scripts/run_demo.py --case hero --no-viser --open
```

Open `http://localhost:8000`. The typed-command hint at the bottom left opens the
command input; `/` does the same. Try:

1. `Show the impact in 3D`
2. `Replay the approach`
3. `Replay the impact in super slow motion`
4. `What if the camera car had been going 30 km/h?`
5. `Who is at fault?`
6. `Fill in the accident statement`
7. `Show the evidence`, then `Verify the evidence`

No GPU or API key is needed for the saved 3D scenes, clip playback, cached report,
counterfactuals, saved accident statement, evidence verification and cached speech.
Online map tiles and fonts still use the internet. New speech transcription,
arguments, segmentation, generation and model reasoning require the appropriate
keys in `.env`; copy `.env.example` and supply your own values for those features.
Uncached spoken replies require an ElevenLabs key.
For new multi-frame enhancement, also install `python -m pip install -e ".[enhance]"`.
The first enhancement downloads the public RAFT and Real-ESRGAN weights if absent.
CPU inference defaults to at most four threads for small crops; override with
`MERCY_INFERENCE_THREADS` when profiling another machine.
Repeated enhancement reuses outputs only when the original crop pixels and all
cached image hashes match. This cache survives extraction on another machine.

The pack also retains the full Nexar license and third-party/film attribution
under `data/licenses/`; source data and derived cases retain their own terms.

All files covered by each case's original signed manifest are included in its data
archive, including server-side intermediates. The private signing key is excluded.
Extraction preserves the original seal; the recipient verifies using its public key.

## Reproduce the film

The shot list and narration are in `docs/demo-storyboard.json`. Revision 6 follows
one London investigation. All ten demonstrated spoken commands use prerecorded
ElevenLabs audio as microphone input, through the application's AudioWorklet,
24 kHz PCM relay, real Muse Voice transcription and real command router. No
transcripts, model responses or result fields are mocked. Processing waits are
shortened with clean cuts. The final soundtrack is ElevenLabs narration, including
the same spoken command, without overlapping application replies.

Visual geolocation was computed automatically during case ingestion, from actual
video keyframes, OCR, scene observations, named-place research and OpenStreetMap.
The voice command reveals that saved automatic analysis. No GPS track or address
was supplied for this case. Editorial cutaways use crops of the same original
keyframes to show the route signs, local business and surroundings. The location
is a proposal with visible uncertainty, not a survey-grade measurement.

Install FFmpeg and FFprobe on PATH. Install Playwright into the ignored tooling
directory, including its recording dependency:

```bash
npm install --prefix data/browser-tests playwright
node data/browser-tests/node_modules/playwright/cli.js install ffmpeg
python scripts/run_demo.py --case london_rearend --port 8001 --no-viser
# In another terminal, with an ElevenLabs key in .env:
python scripts/demo_video.py voices
node scripts/record_demo.cjs
node scripts/record_3d_comparison.cjs
node scripts/record_3d_website.cjs
python scripts/demo_video.py edit
```

The recorder uses installed Microsoft Edge on Windows. It captures separate
1920 × 960 takes, saving real ASR transcripts, application events, screenshots and
explicit source intervals under `deliverables/production/revision6/`. A temporary
production sync frame, excluded from the edit, calibrates each source interval
against the actual recorded frames. Every take is checkpointed and can be rerun
by passing its shot ID to `node scripts/record_demo.cjs 02_plate`.

Playback checks require actual media-time advancement. Map captures wait for
loaded OpenStreetMap tiles. The final 1080p frame reserves space for captions.
The nine-second slot at 2:02–2:11 shows the Evidence page: original frames,
vehicle dossiers, the verified plate and source files. The original/estimated 3D
comparison appears at 1:57–1:59, followed by three seconds of the actual web-app
3D interface at 1:59–2:02. Both use the dashcam camera at T+7.40 s. The navigation,
timeline, estimated-geometry label and original-video inset remain visible in
the interface recording. The 1:48–1:57 quarter-speed section includes the impact.


The speech step reuses previously generated segments by a hash of voice, model
and narration. Re-running the edit makes no ElevenLabs requests. Captions use
the provider's timestamps at the original speech speed. Caption intervals never overlap. The MP4 also
contains a selectable English subtitle track, disabled by default because captions
are already burned in, and embedded chapter markers.

```bash
python -m pytest
python scripts/pack_demo.py --cases hero london_rearend snow_pileup --out deliverables/mercymek-demo-data.zip
python scripts/pack_demo.py --cases cutin_rearend lanechange night_rain vegas_rearend --out deliverables/mercymek-extra-cases.zip
python scripts/prepare_delivery.py package
python scripts/prepare_delivery.py verify
python scripts/demo_video.py verify
```

`prepare_delivery.py verify` extracts the source and both data archives into an
empty directory, verifies every original evidence seal, starts a server without
model or speech API keys, and exercises the saved cases and deterministic commands.
It checks that cached ElevenLabs audio still plays with no credentials or budget.

The video, source and data are prepared locally. Hosting URLs and submission portal
uploads must point to the files after they have actually been uploaded.

## Revision 2 review

The original edit is preserved in `deliverables/production/revision1/`. The revised
production lives in `deliverables/production/revision2/`. The brand is written as
`mercymek` throughout the narration and captions. Repeated fades, artificial camera
wobble and forced voice-speed changes were removed. The hero case remains continuous
through the report, argument, documents and seal before the explicitly introduced
Toronto and London examples.

The hero slow-motion artifact was matched against six decoded source frames. Its
start is source frame 198 (6.6 seconds), with four output frames per source-frame
interval. The output timestamps were corrected from 30.5 to 30 fps without
re-encoding, and the case was resealed. The original metadata and seal, matching
results and corrected metadata are preserved under `production/revision2/data-corrections/`
and `slowmo-alignment.json`. Both the replay counter and main timeline now use the
replay's source time. Slow-motion playback stops at its end instead of looping.

The map-only capture waits for visible OSM tiles and records a fresh map context,
then replaces that section in the edit list. The replay check verifies source time,
normal playback progression and stop-at-end behavior before accepting that take.

## Revision 3 review

The previous edit and its production sources are preserved under
`deliverables/production/revision2/`. The new edit places voice automation first,
spends 50 seconds on video-based location inference and its supporting visual
clues, and uses the London case throughout. Original-speed English ElevenLabs
narration, nonoverlapping subtitles, clean cuts and the spelling `mercymek` are
retained. Per-command ASR and result checks accompany the capture manifest.

The recording pass also exposed two command-routing gaps, now fixed: shortened
scene names such as “silver BMW” select and zoom the saved vehicle, and compound
requests such as “fill in the accident statement and verify the evidence” execute
both actions. View changes use a short opacity transition; removing large blur
and tile-pane filter layers prevents blank map pieces in Chromium captures.
The clean-install check exercises both repaired commands from the packaged source
without API keys. See [the public release review](public-release-review.md) for current
test results.

## Revision 4 review

Revision 3 is preserved as `mercymek-3-minute-demo-v3.mp4`, with its production
sources in `production/revision3/`. Only 2:01–2:10 changes: the spoken “Driver view
and play at half speed” command and reconstruction are replaced by a real spoken
“Show the evidence” command and the Evidence page. That slot has new English
ElevenLabs narration, aligned captions and its own chapter. The other twelve
edited sections and narration segments are reused unchanged. The film remains
exactly 180 seconds, with voice automation and visual geolocation as its focus.

## Revision 5 review

Revision 4 remains preserved with its production sources. Only 1:48–2:01 is
re-edited: nine seconds of actual quarter-speed replay, including the impact,
followed by four seconds pairing the original frame and estimated 3D positions
at T+7.40 s. Both panels come from the same paused application state; the right
panel is the actual WebGL renderer, not generated replacement artwork. Geometry
is labelled as estimated. Evidence stays at 2:01–2:10 and the twelve other edited
sections, including their audio, are reused unchanged. English ElevenLabs speech
and captions for the changed thirteen-second interval are updated at natural speed.

## Revision 6 review

The 3D section now includes the actual web application: two seconds of the
same-frame comparison, then three seconds of the unmodified 3D interface at the
same source time. It is paused in dashcam view, with the original video visible.
Evidence keeps its full nine seconds and moves one second later. One second of
holding time after the what-if explanation is removed to retain exactly 3:00.
All spoken narration is reused without regeneration or speed changes. Twelve
other edited sections, including Evidence, are reused byte for byte.

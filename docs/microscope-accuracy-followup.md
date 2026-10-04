# Microscope presentation follow-up — 2026-09-27

Reviewed commit `96b3ca6` and the live site after the first audit's corrections. The core landscape and four-representation concept are valid educational visualizations. Several remaining claims still give a viewer the wrong impression. The most important work is to make the prominent explanation as careful as the detailed tooltips.

This review changes no application code or recorded data. The original audit remains a historical record, not a description of the corrected version.

## Verified improvements

- The solver is correctly identified as midpoint, with 32 updates and 33 saved boundary states.
- On the live site, Finished mode now shows final-step measurements: waveform, latent and band agreement all display 1.000, with t = 0.
- Reused-score detection is implemented; the replay shows the score whole rather than pretending to replay its writing.
- The acoustic replay phase now admits that it is illustrated, and the replay heading says it is sped up.
- The strongest bass-first / air-last assertions were removed.
- Token IDs are now explained as labels, and latent channels are distinguished from frequency bands.
- Recognized words are identified as potentially erroneous machine output.
- The new 3.6-billion-parameter approximation, 32,768-entry semantic codebook, 25 Hz codes, 64 latent channels, unconstrained ABC sampling and midpoint description are supported by the [official technical report](https://github.com/multimodal-art-projection/YuE/blob/main/docs/technical_report.pdf), especially sections 2–3 and the sampling appendix.
- All 108 Python tests pass. The live focus and stack app scripts fetched during this review match local source byte for byte.

## Remaining presentation problems

### 1. The stack still promises a recorded replay that it does not provide

Locations: `src/audiogen/explore_assets/app/stack.js:9–14, 378–421`; `src/audiogen/explore_assets/index.html:49–52`; `bin/explore.py` publish introduction; `docs/microscope-explorer.md`, stack description.

The public introduction still says everything comes from one generation “saved as it ran,” and the stack card promises the writing “as it happened.” These claims conflict with the reused score and the illustrative acoustic animation. The correction appears only once that animation is underway.

The clock continues to display a fabricated timestamp in the illustrative phase. At slider value 950 the live site showed **“1:34.4 into generation”**, although that position is computed from an invented tail equal to 12% of the last semantic-token timestamp. It is not a recorded time. Sparse checkpoint interpolation also estimates intermediate token counts rather than replaying every observed arrival.

Suggested prominent text: **“Explore the saved representations and an accelerated illustration of the generation stages. Token progress is estimated between recorded checkpoints; the acoustic animation in this view is illustrative.”** During the illustrative phase, replace the clock with **“Acoustic stage — timing not recorded here.”** Preserve the real clock during measured phases, with interpolation disclosed.

### 2. The stack implies that the score alone is successively converted into the song

Locations: `explore_assets/stack.html:16–18`; `explore_assets/app/stack.js:10–14`.

“Each made from the one above it” encourages a simple score → codes → latent → audio interpretation in which earlier inputs disappear. The model also uses the style prompt and lyrics. Its acoustic stage attends to text, score and semantic tokens together. Those inputs matter to a novice trying to understand where the voice, words and arrangement come from. The technical report makes these dependencies explicit in section 2, equation 2 and Figure 3.

The approximately 3.6B main network does generate the score, codes and acoustic latents. “One network ... does all of it” should distinguish the **separately trained audio decoder** used afterward; the four-floor presentation currently includes that output in “all.”

Suggested text: **“The model uses a style prompt and lyrics alongside a musical score to generate sound codes, then a detailed compressed sound representation. A separate decoder turns that representation into audio. These floors show the representations in that process.”** Add the supplied prompt/lyrics as labeled inputs or a short visible sentence. The existing rough-draft metaphor itself is reasonable.

### 3. Color still sounds like proof that a part is finished

Locations: `explore_assets/app/focus.js:12–14, 164–176`; `explore_assets/focus.html:49, 60–62`.

“Frost-teal glitter doesn't match the finished song yet; color does” is still too categorical. The shader blends colors continuously over correlations from 0.45 to 0.99. Color therefore appears before close agreement, and agreement concerns a frequency band's whole-song pattern, not the correctness of a particular visible peak. Disabling settling color also leaves a colored terrain with no such interpretation.

“Locks in” remains the headline, although a band may continue changing substantially after crossing the chosen threshold.

Suggested main explanation: **“With comparison coloring on, warmer colors show a stronger resemblance to the final step's rise-and-fall pattern across the song. They do not mean that a sound is finished.”** Rename the section **“Similarity to the final step, by frequency band.”** Keep the threshold explanation in the detail text.

### 4. “Volume-matched” is not what the audio processing does

Locations: `explore_assets/focus.html:64`; `explore_assets/app/focus.js:25–26`; published introduction; `src/audiogen/microscope.py:191–194, 307`; `src/audiogen/explore.py:475, 509`.

The focus previews are individually peak-normalized to −1 dBFS. Equal maximum peaks do not make recordings equally loud to a listener. For example, the examined listening files have equal −1 dBFS peaks while step-zero Predicted measures −18.64 dBFS RMS and step-zero State measures −16.47 dBFS RMS. This RMS comparison is evidence of unequal signal levels, not a perceptual loudness test. Stack playback also uses the original final WAV as its encoding source rather than the normalized focus preview.

Suggested simple text: **“Preview levels are adjusted for playback; some steps may sound louder than others. Web audio uses AAC compression.”** In optional detail, say **“Each focus preview is normalized to the same peak level.”** Avoid claiming perceptual loudness matching unless it is actually implemented and measured.

### 5. A few labels still contradict the thing being shown

- **Predicted step 0 says “the starting noise.”** Live-browser reproduced. It is actually an endpoint estimate calculated using a model evaluation at the initial state. Use **“Prediction from the initial state”** in Predicted mode (`app/focus.js:803`).
- **Ghost says the height gap is “only the lift.”** It is the artificial 7-unit lift **plus the difference between the displayed surface heights**. Say “The outline is raised for visibility; its separation is not a remaining-error measurement” (`focus.html:47`).
- **“Everything in this panel is measured from the sound” is false.** Latent match is calculated from latent arrays and t is the solver coordinate. Say “These are measurements of the saved latent states and decoded audio” (`focus.html:64`, `app/focus.js:528`).
- **Brightness still promises a simple noise-clearing story.** State's centroid goes approximately 2456 → 2094 → 2409 Hz at steps 0, 12 and 32. It first moves below the final value, then rises again. Define the metric and let its actual curve speak; do not imply that brightness directly measures remaining noise (`app/focus.js:777`).
- **“The sketch settles well before the sound itself comes clean”** infers a perceptual event from latent cosine similarity. Replace it with “Compare how this numerical similarity changes across steps” (`app/focus.js:775`).

### 6. The original microscope document retains the stronger interpretation

Location: `docs/microscope.md:42–50`.

It still says the predicted preview answers what the solver had “already decided,” and describes intermediate states as essentially the final fading in through noise. The linear noise/data mixture in the training construction does not establish that every inference state follows the straight line connecting its initial noise and eventual generated endpoint; decoding is also nonlinear.

Suggested wording: **“State decodes the saved intermediate latent. Predicted extrapolates the currently estimated velocity to the endpoint; it is a preview, not a record of an irreversible decision.”** This is consistent with the improved app glossary.

### 7. The visible step can get ahead of the audible step

Locations: `src/audiogen/explore_assets/app/focus.js:468–481, 733, 875–881`; `src/audiogen/explore_assets/focus.html:24`.

The new guard correctly disables the listening-game buttons during an audio switch. However, the terrain and measurements immediately follow the selected step while the previous audio continues until the new buffer is decoded. There is no visible switching status. Preloading encoded files does not eliminate subsequent decoding delays.

Reproduced against the live site with a controlled 3.5-second delay on subsequent `AudioContext.decodeAudioData` calls: while playing step 0, selecting step 10 immediately showed **“31% of the steps done … t 0.69”** while `audio.loadedStep` remained **0**. The game buttons were correctly disabled. This is an induced slow-decoding case, not a measurement of typical delay, but it demonstrates a real mismatch the app permits.

For a learning app this matters: a listener can attribute the previous step's sound to the newly displayed state. Keep the displayed state synchronized with the audible state during playback, or show an explicit **“Loading step 10; playing step 0”** status and distinguish the pending selection. Replace “nothing waits once it plays” with wording that describes the actual preloading guarantee.

## An accurate short explanation for a newcomer

> The model uses lyrics, a style description and a musical score to build a compressed representation of a song. During its acoustic stage, that representation starts as random values and is updated over 32 steps. We decoded the saved states so you can hear them and see their frequency patterns. Predicted is a quick estimate of the endpoint from a selected step. The colors compare patterns with the final recording; they do not tell us which instruments the model has finished. The Stack shows the representations and an accelerated illustration of the stages.

The visualizations do not need to become mathematically literal to be useful. The remaining priority is consistent, visible labeling of **recorded state**, **derived estimate**, **illustrative animation**, and **retrospective similarity**. A novice should not have to find a tooltip to discover that a prominent statement was too strong.

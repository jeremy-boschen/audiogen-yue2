# Done

Measurements and what they ruled out. Git log has the what; this keeps the numbers.

## 2026-09-26 — generation microscope, first milestone

Fixture: Burn It Down, seed 777, 200 s, `bin/microscope.py capture` in `.venv-dev`
(the lockfile's packages, engine editable from ../YuE). Runs under
`../audiogen/output/microscope-runs/`.

- **Observation is inert.** Pinned engine (7ba7e09), new engine with no observer, and full
  capture (33 states + velocities copied to host every step, all decoded afterwards):
  abc f635635350e6971c, semantic 13949b457fa5bbef, latent e04c2e81c24fd269,
  pcm fb52a658072eebc4, decoded FLAC samples identical. 5000 tokens, so both MPS gates
  (1024, 4096) crossed. The last ODE state equals the returned latent; decoding it
  reproduces the PCM hash.
- **ComfyUI canary** rebuilt from audiogen-comfyui 66c768b: both steps bit-identical.
- **ABC:** tempo/meter/unit in the first 50 tokens, key by 100; the score is then written
  strictly left to right (vocal notes 4% / 26% / 66% / 100% at 400 / 800 / 1600 / 2607 tokens).
- **ODE, predicted final vs finished take** (waveform corr): 0.38 step 0, 0.83 step 4,
  0.92 step 8. Per band: bass >0.93 by step 4, mids 0.91 by 12, >2.5 kHz 0.95 by 24.
  Envelope (phase-insensitive): 0.89 at step 0, 0.97 at step 8.
- **Same tokens, other noise** (4 seeds): waveform corr -0.03..0.11 but envelope 0.960;
  other seeds' takes score 0.38-0.46. Noise changes phase and detail, not what plays when.
  Ruled out: reading waveform correlation as musical sameness.
- **N-step solves:** 12-64 steps land within 0.4% of the 32-step waveform (corr >= 0.996);
  1 step 0.74; 48 and 64 steps are further from 32 than 24 is.
- **Semantic prefixes** render the same span of the take (envelope 0.97-0.998) but not
  identically. CPU noise draws are prefix-stable (verified), so noise is ruled out.
- **Envelope metric bug:** 2 of 48 log bands held no FFT bin and inflated every
  correlation (noise seeds read 0.987, pure noise's low band 0.94). Fixed by dropping them.
- **Seed study (8 seeds):** seed 777 reproduces the baseline exactly. Every seed plans
  220-282 s of score against a 200 s budget; all are truncated.

## 2026-09-26 — supplied scores take the planner's form

- **Planned scores all end in exactly one newline**, and `request_for` stripped it, so a
  planned score supplied back as text changed its last ABC token and the whole take
  (Burn It Down s777: semantic f501d690 vs 13949b45). `render.canonical_score` now trims
  and restores that one newline. Every planned score in `output/**/*.take.zip` that
  re-encodes at all re-encodes exactly this way (16 distinct).
- **Text cannot hold every score.** 8 cheap_looks scores (abc temperature 1.3) were written
  in non-canonical BPE splits (`C`+`FA` where encoding gives `CF`+`A`); no text rule
  reproduces them. Only saved plan tokens (`--plan-from`, a take bundle) are exact.
- Songs and studio takes from before this carry `"normalize_score": false`, which keeps
  the old trimmed form, so they still reproduce from their text.

## 2026-09-26 — fixed-ABC study and the Slow Down fixture

- **Fixed ABC, 8 seeds** (`studies/burn_it_down-fixed_abc.json`, plan tokens held with
  `--plan-from`): every run has abc f635635350e6971c; seed 777 reproduces the baseline
  (semantic 13949b457fa5bbef); the other 7 give 7 different semantic streams. All stop at
  the 5000-token budget, as in the seed study.
- **Slow Down take 2 is the explorer's fixture** (`20260926-slow_down-s4417-take2/on-cpu`),
  captured from studio take #233's own bundle with `--plan-from`: semantic
  03afb4c03992cbe5 and latent ea116c64d81f212e equal the bundle's, and the decoded FLAC
  samples are identical (4456320 frames). It ends on its own end token (2322 tokens, 92.84 s).

## 2026-09-26 — the take's own noise, transformed (Slow Down take 2)

`bin/microscope.py noise-variants`, engine b576b12 (`synthesize(noise=)`). Signal stats
against the take; not listening results.

- **same** (untransformed, supplied back): pcm 0f1c1ac604941d18 = the take. The supplied-noise
  path changes nothing by itself.
- **flip** (every sign inverted): waveform corr -0.505 (low-mid -0.75, mid -0.65), envelope
  0.924. Much of the fine waveform follows the noise's sign: flipping the noise largely flips
  the output's polarity, which the ear does not hear as a difference.
- **reverse** (frames in reverse time order): waveform corr +0.08, like an unrelated noise
  draw; envelope 0.939.

## 2026-09-26 — the latent's channels as effects; starting from another song

Slow Down take 2, excerpt-based. Signal statistics unless marked "by ear".

- **Latent channels act like character effects** when nudged before decoding
  (`latent_mixer`, `latent_atlas`, `latent_probe`, `latent_freqmap`). By ear: ch0 "old radio",
  ch26 stereo separation. Measured: ch0 band-pass + distortion +4 dB + narrower stereo;
  ch1 noise +4 dB; ch15 L/R correlation +/-0.70 (width); ch26 hard-left tone leaks +26 dB
  (spread); ch34 decay +509/-282 ms, crest -5/+5 dB, big 300 Hz-3 kHz swing; ch38 noise
  +14 dB; ch55 up to +19/-30 dB above 8 kHz. No channel moves pitch (0 cents, all 64).
- **56 of 64 channels change a pure tone by < 3 dB at every pitch** 50 Hz-10 kHz; the latent's
  variance is spread thinly (first principal direction 4.8%, 16 directions 39%). The quiet
  channels look like distributed detail, not hidden register-specific knobs.
- **Decoder first-layer weights predict impact:** rank correlation 0.63 with the measured
  impact; same top five (34, 55, 15, 22, 26). They say how much, not what.
- **EQ atlas on a song mistakes width for EQ:** it measures the mono sum, so ch15's width
  change read as "presence" there; on the mono sweep its EQ change is small.
- **Starting the solve from another song** (Cannons, Bad Dream, blended with the take's noise
  at round k): envelope vs take 0.90 / 0.60 / 0.33 at rounds 8 / 16 / 24, vs Bad Dream
  0.33 / 0.71 / 0.91. By ear: "doesn't do as much as I thought, and eventually just turns
  into Bad Dream". The blend carries content, not just sound.
- **Principal-direction knobs: nothing by ear.** Top 12 directions of 87 takes' latents (59 of
  them cheap_looks), offered in the mixer: "didn't notice anything that sounded good". The
  strongest are mixes of the single-channel knobs. Now behind `latent_mixer.py --directions`.
- **Matching one reference's channel statistics** (Bad Dream, 0-150%): by ear "isn't giving us
  much, at least not with only 1 song". Measured at 100%: +6 dB louder, centroid 314 -> 471 Hz.
- **The decoder's context is about one second.** One latent frame changed (+3 std, all
  channels) alters the audio from 12 frames before to 12 after (0.48 s each side; >1% of peak
  within 0.28 s). Decoding in 1 s pieces with 0.64 s of context each side matches a whole-take
  decode to -122 dB (float rounding); 4 s pieces with that context are bit-identical; no context
  gives -18 dB of error at the joins. A streaming mixer would hear the same audio, sooner.

## Legacy decoder vs default on the same latent (2026-09-26)

Slow Down take 2's final latent was decoded by YuE2-Vae and YuE2-Vae-legacy (sha256 b6d28362…, `models/YuE2-Vae-legacy` symlink). The default decode is bit-identical to the take.
- Level: −16.0 vs −16.1 dB. Centroid: 311 vs 314 Hz.
- Band correlation: bass, low-mid and mid 1.00; presence 0.77; air 0.30.
- The difference is −21.5 dB relative to the signal.

The tone, including the `/` tilt, comes from the latent. The two decoders differ only in fine detail above ~4 kHz. A/B page: RUN/on-cpu/decoders/index.html.

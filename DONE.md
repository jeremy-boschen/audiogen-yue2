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

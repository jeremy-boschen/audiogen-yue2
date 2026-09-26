# Next

Open work only. Each item has a checkable "done when"; finished items move to DONE.md.

## Now

- **Commit trailers say `opus-5`, should say `opus-5.5`.** 30 unpushed commits from
  2026-09-26 across YuE, audiogen-yue2, spectrum-local and audiogen. Awaiting the user's
  go to rewrite messages only (YuE's SHA moves, so `env/pins.env` moves with it).
  Done when: every 2026-09-26 trailer reads `opus-5.5` and each rewritten branch's
  `rev-parse ^{tree}` equals its tree before the rewrite.
- **Push the engine commit** (`YuE` local-fixes, "Let a caller watch the acoustic solve").
  `env/pins.env` already pins it, so `./setup.sh` without `--dev` fails until it is pushed.
  Done when: `git ls-remote origin local-fixes` shows the pinned SHA.
- **Listen and annotate the baseline ODE.** `bin/microscope.py annotate` / the run's
  `index.html` checklist. Done when: `analysis/timeline.md` of
  `20260926-burn_it_down-s777-baseline/on-cpu` has vocal_present, vocal_melody_recognizable
  and words_intelligible filled for both the ODE state and the predicted final.
## Soon

- **Burn It Down's budget truncates every seed.** All 8 seeds plan 220-282 s of score
  against a 200 s budget. Done when: the user decides the budget, and a seed-study rerun
  has every take end on the model's own end token (`semantic tokens < max_tokens`).
- **Chorus phrases do not pair with lyric lines.** Score-line and rest-gap segmentation both
  miss the chorus shape (hooks repeat within a line). Done when: a syllable-to-note
  alignment pairs the choruses of the baseline, Sunday Kitchen and Slow Down, or reports why not.
- **Correlate intelligibility with notes/syllable.** Needs manual phrase annotations across
  the seed study. Done when: `seed-summary.json` rows carry manual_intelligibility for
  every paired phrase.
- **Separate "future tokens" from "position offset"** in semantic prefix renders (prefix
  audio differs from the full take's head; noise is ruled out). Done when: a render with the
  full AR prefix but only N frames solved, or an equivalent, attributes the difference.
- **Microscope runs in the Soundroom.** The studio lists only takes it rendered; runs are
  opened as local pages. Done when: the user decides whether they belong there.

## Research (banked)

- **Effect knobs on a finished take (Soundroom).** Latent channels behave like character
  effects when nudged before decoding (`bin/latent_mixer.py`, `bin/latent_atlas.py`,
  `bin/latent_probe.py`): e.g. ch0 band-pass + distortion + narrower stereo ("old radio" by
  ear), ch1 raises the noise floor ("grain"), ch55 air/brightness, ch34 body, ch15 presence,
  ch26 stereo spread (+26 dB left-to-right leak, wider on decorrelated noise). A take-level
  Sound panel: named knobs, live looped preview with A/B, "keep" decodes the whole take with
  the settings as a new version (original untouched, settings stored with it). Before building:
  confirm the map on a second and third song, and set each knob's safe range from the probe.
  Done when: the map is confirmed across songs and the user has chosen the knob set.

- **Start a take from another song's sound (next test after the probe kit).** The VAE ships
  its encoder (`modeling_vae.py` `encode`, `encoder.*` weights in YuE2-Vae), so any recording
  can become a latent. Encode a known song (Cannons, in `~/Music/YouTube Downloads/Mosac`),
  then solve Slow Down take 2's tokens from a partial blend of it with noise (start at round k
  of 32 instead of 0, via `synthesize(noise=)` or a start-step hook) at a few blend amounts.
  Question: does the reference's tone/weight ("mastered the same way") carry over, and when
  does its own rhythm start ghosting through? Done when: the user has heard the blends.

- **Tone tilt on a finished take.** A post-decode EQ (tilt / high shelf / low cut) applied to
  a take without re-rendering, so the same performance can lean `/` (warm) or `\` (bright).
  Plain mixing, not the model; the question is whether it belongs in the Soundroom as a
  per-take control, and whether it is offered before or after the latent-direction idea
  (nudging the finished latent along a learned "brightness" direction before decoding).
  Done when: the user has heard an A/B of one take tilted both ways and decided.

## Open questions

- Is waveform correlation of the predicted final (0.83 at step 4) matched by ear? Signal
  says bass settles by step ~4, mids ~12, >2.5 kHz by ~24; listening decides.

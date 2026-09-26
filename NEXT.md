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
- **Fixed-ABC study** (running, holding the baseline's exact plan tokens). Done when:
  `fixed_abc-summary.json` exists and seed 777's run has the baseline's semantic hash.

## Soon

- **Burn It Down's budget truncates every seed.** All 8 seeds plan 220-282 s of score
  against a 200 s budget. Done when: the user decides the budget, and a seed-study rerun
  has every take end on the model's own end token (`semantic tokens < max_tokens`).
- **A supplied score is not the planned score.** `request_for` strips the score text, which
  drops the planned score's trailing newline and changes its last ABC token (2607 ids, first
  difference at 2606); from seed 777 that gives a different take (semantic f501d690 vs
  13949b45). Changing the strip alters every existing song with a supplied score.abc, so it
  is the user's call. Done when: decided, and if changed, cheap_arcade re-renders are checked.
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

## Open questions

- Is waveform correlation of the predicted final (0.83 at step 4) matched by ear? Signal
  says bass settles by step ~4, mids ~12, >2.5 kHz by ~24; listening decides.

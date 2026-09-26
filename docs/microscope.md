# The generation microscope

`bin/microscope.py` renders one take the normal way (`render.render_step`) and
watches it being made:

| stage | observed through | what a checkpoint is |
|---|---|---|
| ABC plan | the engine's `on_token` | the first N tokens of the final plan |
| semantic | the engine's `on_token` | the first N codec tokens (CODEC_OFFSET removed) |
| acoustic ODE | the fork's `on_step` (`yue2.nar.FlowStep`) | the solver state after step N, plus the velocity it evaluated there |

Token streams are autoregressive: nothing emitted is ever revised, so a prefix
checkpoint is a slice of the final stream. The callbacks add arrival times and a
check that the stream observed is the stream returned, nothing more. The ODE is
the stage whose intermediate states exist nowhere else.

## The rule: observation must not change the take

Every claim from a microscope run rests on an `--off` twin being bit-identical:

    bin/microscope.py capture burn_it_down --run $R/off --off
    bin/microscope.py capture burn_it_down --run $R/on
    bin/microscope.py compare $R/off $R/on           # abc, semantic, latent, pcm + decoded FLAC

How the engine keeps it inert: `on_step` is reported only where the solver
already has the tensors (no added model call), the solver never writes a
tensor after reporting it, and with no observer nothing is passed at all.
How this module keeps it inert: callbacks only append; each state and velocity
is copied to host memory as reported (`--capture cpu`, the default); all
decoding, analysis and file I/O happen after the take is finished and its own
PCM is decoded. `--capture reference` holds the device tensors instead (no
copies, but their memory stays allocated during the solve); use it only after
its own `compare`.

A capture also checks itself (`analysis/capture.json` → `integrity`): the
observed token streams equal the returned ones, the last ODE state equals the
returned latent, and decoding that state reproduces the take's PCM hash.

## Two views of the ODE

- `flow/chunk_000/step_SS.npy` and `flow_audio_raw/` — the solver state
  itself. On a flow path the state at step k is close to a blend of noise and
  the final, so it mostly shows the final fading in through noise.
- `flow/chunk_000/predicted_SS.npy` and `flow_predicted_audio_raw/` — the
  solver's estimate of the finished latent at step k, `x - t*v`, derived in
  FP32 on the host from the reported state and velocity (the audio analogue of
  a diffusion x0 preview). This is the view that answers "what had the solver
  already decided at step k". It is an observation of the solve, not a value
  the solver computed.

Raw audio is float WAV, never normalised. `*_listening/` copies are peak
normalised to -1 dBFS for listening and are never the measured file.

## Run layout

    <run>/
      metadata.json            invocation, repo commits, engine install, machine, engine config, provenance
      prompt/                  style, lyrics, request
      take/                    the engine's native artifacts (score, tokens, latent, FLAC, receipts)
      baseline/hashes.json     stage hashes, the same ones bin/render.py prints
      plan/                    tokens/NNNNNN.npy, abc/NNNNNN.raw.txt (authoritative) and .valid.abc (derived), analysis.json
      semantic/                tokens/, analysis.json, prefix_audio/ (from `prefixes`)
      flow/chunk_000/          step_SS.npy, predicted_SS.npy, metrics.json
      flow_audio_raw/ flow_predicted_audio_raw/ (+ _listening copies)
      final/                   latent.npy, audio.wav
      ode_steps/               from `ode-steps`
      analysis/                alignment.{json,txt}, annotations.json, timeline.{json,md}, metrics.csv, capture.json

Multi-chunk songs keep one `chunk_NNN/` per acoustic chunk, and each chunk's
metrics record its song frame range and pinned lead frames. A chunk is decoded
alone, which is not how the song's own decode treats its edges.

## Listening

Annotate after listening; the timeline takes the first step judged true:

    bin/microscope.py annotate $R --step 6 vocal_present=true beat_recognizable=true
    bin/microscope.py annotate $R --step 3 --predicted vocal_present=true
    bin/microscope.py annotate $R --phrase verse.1.2 "garbled: 'cheapest' sung as 'chee-ee'"
    bin/microscope.py timeline $R

Metric thresholds in the timeline (correlation with the final, latent cosine)
locate where things change. They are not listening results.

## Melody against lyrics

`analysis/alignment.txt` pairs each score section with its lyric section by
label and order, and each phrase with a lyric line when the counts agree —
first by score line (YuE2 usually writes one lyric line per `V: Vocal` line),
then by rest gaps. When neither pairs, the section says so rather than
guessing. Syllables are a vowel-group heuristic and approximate; the bins
(<0.8 … >2.0 notes/syllable) group phrases for reading and are not rules.

## Further experiments

    bin/microscope.py prefixes $R --tokens 500,1000,2000,4000   # each prefix solved alone, same seed, no padding
    bin/microscope.py ode-steps $R --steps 1,2,4,8,16,32,64     # final outputs of N-step solves, tokens fixed
    bin/microscope.py study studies/seed.json                   # see cmd_study for the manifest shape

Step 8 of a 32-step solve and the output of an 8-step solve are different
states; `flow/` holds the first, `ode_steps/` the second.

# Line-for-line: the engine vs the ComfyUI stack that made Baseline01

> Historical investigation. See [PARITY_PROGRESS.md](PARITY_PROGRESS.md) for
> subsequent measurements and corrections: the original prefix comparison
> missed Plan-node score trimming, and runtime RMSNorm was also monkeypatched.
> Those findings invalidate the conclusions that the context was identical and
> that Metal attention had been ruled out. The original riff and full recorded
> continuation chain now reproduce exactly; see [PARITY_ACCEPTANCE.md](PARITY_ACCEPTANCE.md)
> for the remaining validation gates.

## What this is, and why

The album *Forgives at Five* was made in ComfyUI, through the `ComfyUI-FL-YuE2`
node pack. That environment reproduces itself exactly — `ENV_album.md` records
the launch flags, the pinned commits and the recipe, and the whole four-step
lineage from the 45s riff to `burn_01_master` regenerates bit-identical.

But it reproduces only as long as ComfyUI does. The pack is a wrapper around
YuE2 carrying its own copies of the model code, and the node graph is not a
thing you can version, diff, or run from a shell. So the work moved to a second
repository, `audiogen-yue2`, built directly on a fork of the official engine
(`YuE/src/yue2/`) with a CLI and song manifests. That stack is where new songs
are supposed to be made.

The problem: **the two stacks do not produce the same audio from the same
inputs.** Same weights, same seed, same prompt, same torch build — different
song. Until that is understood, the new stack cannot be trusted to continue the
album's lineage, and every existing take is stranded in ComfyUI.

So the task is a bisect: find every place the two code paths differ, change one
variable at a time, and measure. The target is "Baseline01" — the 45s riff at
the head of the master's lineage, seed 777, which is short enough to iterate on
and sits at the root of everything else.

Two rules govern it, both set by the user after I broke them:

- **One change per variable.** A shim that bundles N behaviours counts as N
  changes and gets split into rungs. A measurement attributed to a compound
  patch is not a single-variable result.
- **Call the real op, or reimplement it from the real source.** No ComfyUI
  imports in this repo; no reconstructing an op from its docstring. A borrowed
  op makes a row that cannot be rebuilt from this repository alone.

The user's ear is the final authority on anything musical. The numbers here
describe signal, not music, and a re-roll is not a comparison.

## How far this has got

**Not reproduced.** The riff still differs.

One real cause was found and fixed: **#1, the RMSNorm rounding** (`norm_c`). It
was worth 1.69% → 18.04% on its own and it is the single largest known
contributor.

With it applied, the two forward passes agree on **97.60%** of per-step argmaxes
when fed identical context, and every one of the 27 disagreements is a near-tie
decided at one bfloat16 ulp. But one flipped near-tie at step 126 cascades, so
free-running agreement is **19.29%** and the audio is a different take.

Eleven differences have been found. Nine are applied or eliminated. Two are real
and still unmatched: **#10** (`--reserve-vram`, live in ComfyUI, dropped on our
MPS) and **#11** (a Metal attention kernel that a *different* node pack
monkeypatches into `torch.nn.functional` at ComfyUI startup, absent from our
stack entirely).

#11 was found this session and is the most recent dead end worth knowing about:
it is unambiguously load-bearing — switching it off inside ComfyUI changes 80% of
ComfyUI's own greedy tokens — but adding it to our stack moves us *further* from
the target, not closer. It is a genuine entry in the diff and not the cause of
the residue.

**The 2.40% residue is unexplained.** Every result so far compares outputs and
reasons backwards; nothing has yet established that the tensors entering
attention are identical at a given decode step. That is the next move, and it is
measurement, not another candidate.

---

## How to read the tables

Baseline01 is the 45s riff, seed 777, rendered by
`ComfyUI/custom_nodes/ComfyUI-FL-YuE2/` (`run04_comfyui_riff_today`). In every
table below the left column is that pack and the right is `YuE/src/yue2/`.

**Every row carries a yes/no: did I verify it by running a measurement?**
"Yes" means a test in this project directly establishes the claim. "No" means it
rests on reading source, on inference, or on a weaker metric than the claim needs
— regardless of how confident it looks. No partial credit.

Model shape: 28 layers, hidden 2048, head_dim 128, 16 query heads, 8 KV heads
(GQA factor 2), rope_theta 1e6, `tie_word_embeddings: false`.

The measurement that matters: **teacher-forced greedy**. Our model is fed
ComfyUI's own greedy token stream and we take the argmax over `distribution()`
output — band mask, min_tokens rule and repetition penalty over the last 50 —
which is exactly what both stacks argmax. Token identity at temperature 1.0,
used for most of this investigation, is insensitive to sub-ulp logit changes and
several early "inert" verdicts rest on it alone. Those are marked No.

---

## Differences found

| # | pack | engine | what differs | verdict | verified by test? |
|---|---|---|---|---|---|
| 1 | `comfy/ops.py:687` `F.rms_norm(x, shape, w, eps)` | `modeling_yue2.py:132` `x * rsqrt(x.float().pow(2).mean(-1)+eps).to(x.dtype) * self.weight` | three roundings vs one | **REAL — 1.69% → 18.04%** | **Yes** |
| 2 | `model.py:53-62` `comfy_kitchen.rms_rope_split_half` | `modeling_yue2.py:185-188` norm, then `_apply_rotary` on each | one fused call vs two ops | subsumed by #1 — bit-identical tokens/latents/audio once `norm_c` is on | **Yes** |
| 3a | `model.py:24` explicit −inf float mask | `modeling_yue2.py:213` `is_causal=True` | mask form | inert | **No** — temp-1.0 token identity only |
| 3b | `model.py:18` causal prefill in 256-query blocks | one call | blocking | inert | **No** — temp-1.0 token identity only |
| 3c | `model.py:12-28` the pack's whole AR attention path | `modeling_yue2.py:191-213` | mask rank, reshape order, GQA expansion | inert | **Yes** — greedy teacher-forced, 27/1125 unchanged |
| 4 | `ops.py:56-61` expand KV only when a mask exists, else `enable_gqa=True` | `modeling_yue2.py:23-28` always `repeat_interleave` on MPS | who expands GQA on decode | inert | **Yes** — greedy teacher-forced, 27/1125 unchanged |
| 5 | `sampling.py:57,96` `prior=carried`; `runtime.py:125` | no `prior` parameter | penalty window across steps | inert on a first step — `runtime.py:107` makes `carried == []` | **No** — read, never exercised; needs a chained render |
| 6 | no decorator; `execution.py:751` wraps all nodes | `sampling.py:73` `@torch.inference_mode()` | — | not a difference; both run under inference mode | **No** — read `execution.py`; the shim OOMed 3/3 at token 70 and never produced a comparison |
| 7 | `sampling.py:125` no `logits_to_keep` | `sampling.py:156` `logits_to_keep=1` | slice vs full on decode | inert | **Yes** — greedy teacher-forced, 27/1125 unchanged |
| 8 | `model.py:156` `lm_head(model.norm(x[:, -k:]))` — slice then norm | `modeling_yue2.py:443` norm then slice at `:554` | norm over 2149 rows vs 1 | inert | **Yes** — greedy teacher-forced, 27/1125 unchanged |
| 9 | `model.py:119-131` cache `[B,T,H,D]`, `torch.empty`, `.copy_` | `modeling_yue2.py:371-393` cache `[B,H,T,D]`, `torch.zeros`, slice-assign | SDPA input strides | inert | **Yes** — greedy teacher-forced, 27/1125 unchanged; also unchanged when combined with #11 |
| 10 | `--reserve-vram 8` shapes ComfyUI's model management | `pipeline.py:163` gates the budget on `device.type == "cuda"`, so it is dropped on MPS | the flag string matches, the effect does not | **UNMATCHED** | **No** — read the gate; never re-ran ComfyUI without the flag |
| 11 | `AppleSilicon-FP8/__init__.py:98` installs `_patches/flash_attn_mtl.py`, which rebinds `torch.nn.functional.scaled_dot_product_attention` to a Metal kernel (`metal_flash_attn/sdpa.py:176-178`) | stock `F.scaled_dot_product_attention` | **the op itself, on every decode step** | **REAL and unmatched — but not the cause of our residue** | **Yes** |

### #11 in detail

This op is in neither yue2 tree. It is installed by a *different* node pack, at
every ComfyUI startup, with no launch flag. Both stacks call
`F.scaled_dot_product_attention`; in ComfyUI's process that name does not mean
torch's op. A source diff of `model.py` against `modeling_yue2.py` — which is what
I did for weeks — cannot see it.

The gate (`metal_flash_attn/sdpa.py:33-73`) on the riff, **measured**:

| call | shapes | gate result | routed to |
|---|---|---|---|
| ComfyUI prefill | mask present (`model.py:24`) | `(False, 'attn_mask')` | stock SDPA |
| engine prefill | `is_causal=True`, no mask | `(True, 'fast-tier')` | *would* route — a call ComfyUI leaves alone |
| decode, ×1125 ×28 layers | q `[1,16,1,128]`, k/v `[1,8,Lk,128]`, Lk 2149→3274 | `(True, 'fast-tier')`, tier `v2_bf16` | **Metal kernel** |

The gate reads `max(Lq, Lk)`, not `Lq`, so a 1-token query against a 2149-token
cache qualifies from the very first decode step. Kernel vs stock on those shapes:
72–73% of output elements differ, max abs 0.00098.

Load-bearing, measured inside ComfyUI itself: relaunching with
`MTLFLASHATTN_SDPA=off` and nothing else changed moves **901/1125 = 80.09%** of
ComfyUI's own greedy tokens, first difference at token 84. `ENV_album.md:57-59`
already said "it is part of the sound"; that is now a measured statement.

## The two experiments that closed #11 as a cause

Teacher-forced greedy, per-step argmax agreement against ComfyUI's stream:

| our stack | ComfyUI target | mismatch | first at |
|---|---|---|---|
| `norm_c` | kernel **ON** (Baseline01 conditions) | 27/1125 = **2.40%** | 126 |
| `norm_c` | kernel **OFF** | 29/1125 = 2.58% | 84 |
| `norm_c` + `mtl_b` | kernel **ON** | 35/1125 = **3.11%** | 36 |
| `norm_c` + `mtl_b` + `cache` | kernel ON | 35/1125 = 3.11% | 36 |
| `norm_c` + `cache` | kernel ON | 27/1125 = 2.40% | 126 |

Taking the kernel away from ComfyUI moves ComfyUI *away* from us. Giving the
kernel to our stack moves us *away* from ComfyUI. Both directions are worse, so
our stock-SDPA path sits numerically between the two and the 2.40% residue has a
different cause. Verified by test: **Yes**.

The first-mismatch index is the sharper number: 126 → 36 means the perturbation
reached a near-tie 90 steps earlier, i.e. the change was bigger. The percentages
understate it because after the first flip both streams are still force-fed the
same tokens.

**The reasoning error worth recording:** I found a difference that was real and
large (80%) and treated "real and large" as "therefore the cause", without
testing the direction. Size tells you a thing matters; it does not tell you it
matters *toward* the target.

## Verified identical, not assumed

| thing | how | verified by test? |
|---|---|---|
| prefix / prompt | 2149 tokens, token for token, rebuilt with the **pack's own** `protocol.py` + tokenizer from our `request.json` | **Yes** |
| torch build | same git hash `08187d9e0fba026dc8217405802ab5381dc88d90`, Python 3.13.15 in both venvs | **Yes** |
| packages | 9 version deltas, each swapped and re-rendered — all inert (`audiogen/output/pkg_align.log`) | **Yes** |
| VAE `weight_norm` | old `weight_norm` and `parametrizations.weight_norm` both call `torch._weight_norm`, measured bit-identical on CPU and MPS | **Yes** |
| kernel GQA layout | `flash_attn_forward` returns bit-identical output for 16/8 GQA and 16/16 pre-expanded k/v at the riff's shapes — so #11's shim does not smuggle in #4 | **Yes** |
| MPS allocator state | a deliberately dirtied allocator (200 throwaway 4096² bf16 matmuls first) produced bit-identical output to a clean one: `semantic 0f52da4fbea39e55` both | **Yes** |
| `protocol.py` | diffed; engine adds only `rng_device`, `id` and validators | **No** — read |
| tokenizer | diffed; engine adds only `from_pretrained` / `save_pretrained` | **No** — read |
| `MLP.forward` | `down_proj(F.silu(gate_proj(x)) * up_proj(x))`, character for character | **No** — read |
| `DecoderLayer` AR branch | same two residual adds | **No** — read |
| `RotaryEmbedding` | same fp32 formula; engine caches `_inv_freq`, pack recomputes | **No** — read |
| `position_ids` | both `arange(past_len, past_len + seq_len)[None]` | **No** — read |
| AR weights | one BF16 checkpoint, `assign=True` in both, no cast | **No** — read |
| comfy `RMSNorm` | `comfy/ops.py:680-694` is `F.rms_norm(...)`, identical to `norm_c` | **No** — read |
| comfy `Linear` | `model_management.py:1543-1554` `cast_to` returns the weight unchanged when dtype and device already match | **No** — read |
| comfy `Embedding` | `ops.py:790-793` sets `out_dtype = None` for a bf16/fp16 weight, so no int64 cast happens | **No** — read |

## Method findings

| finding | verified by test? |
|---|---|
| Teacher-forced greedy is the sensitive metric; temperature-1.0 token identity cannot see sub-ulp logit changes | **Yes** — six shim combinations that looked inert at temp 1.0 were re-measured under greedy |
| An early teacher-forcing run reported 44.71% because it argmaxed raw logits and skipped the repetition penalty both stacks apply first. The tell was a median gap of exactly −1.0000 | **Yes** |
| An early bisect recorded shims 1+2 at a mean logit difference of 0.0209 vs 0.0169 unshimmed — worse than none. That run had torch mismatched between the stacks. **Do not re-quote 0.0209.** | **Yes** |
| A source diff cannot see a monkeypatch. `F.scaled_dot_product_attention` means different functions in the two processes (#11). Verify bindings, not call sites | **Yes** |

## Where that leaves it

Eleven differences found. Nine are applied or eliminated. Two are real and
unmatched: **#10** (`--reserve-vram`, inert on our MPS, live in ComfyUI) and
**#11** (the Metal attention kernel, absent from our stack).

With `norm_c` applied, the two forward passes agree on 97.60% of per-step
argmaxes under identical context, and every disagreement is a near-tie: median
logit gap −0.0625, which is one bfloat16 ulp at that magnitude, worst −0.1250,
13 of 27 within 0.05. One flipped near-tie at step 126 cascades and free-running
agreement collapses to 19.29%.

**The 2.40% residue is still unexplained.** Every result so far compares outputs
and infers backwards. Nothing has established that our q/k/v entering attention
are identical to ComfyUI's at a given decode step.

The next move is therefore not another candidate: it is to dump the actual
tensors at one decode step from both processes, layer by layer, and find the
first op where they part. Until that is done, any named cause is a guess.

Untested leads, in the order they deserve attention:

1. **#10** — re-run ComfyUI's greedy riff without `--reserve-vram` and compare to
   the take it produced with it. Requires restarting the server.
2. **NAR/acoustic stage** — the pack's `nar.py:79-80` routes through the same
   `attention()` as the AR path, so ComfyUI's acoustic stage also reaches the
   Metal kernel on its non-causal calls, while our `nar.py:124` calls
   `F.scaled_dot_product_attention` directly. This affects audio, not tokens, and
   `mtl_b` does not reach it. Unmeasured.

## Shims for the differences above

`src/audiogen/shims.py`. Ladders are mutually exclusive; a shim that bundles N
behaviours counts as N changes and is split into rungs.

    norm_a/b/c    #1, three rungs, ending at the op ComfyUI calls
    qkrope        #2
    attn_a/b/c    #3, three rungs
    gqa           #4
    nograd        #6      OOMs; no result
    alllogits     #7
    lastnorm      #8
    cache         #9
    mtl_a         #11, the real patch installed globally as ComfyUI's startup does
    mtl_b         #11, narrowed to the calls ComfyUI actually dispatches

`mtl_a`/`mtl_b`/`gqa` are one exclusive ladder — all three replace
`modeling_yue2.sdpa`.

Nothing here imports from ComfyUI. `mtlflashattn==0.2.0` is a standalone pip
package — the same one the FP8 pack calls — now installed in the yue2 venv but
**not yet recorded in `env/pins.env` or `env/requirements.in`**.

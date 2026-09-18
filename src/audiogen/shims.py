"""Opt-in patches that make the engine compute an op the way ComfyUI does.

The album was rendered by the ComfyUI FL-YuE2 node pack. Three op-level
differences were bisected between that pack and the engine, and each one is
reproduced here as a patch that can be switched on and off for a single run, so
two takes can be compared by ear without rebuilding anything.

These are monkeypatches on purpose. They belong in the engine eventually --
they are model-internal, not conventions of ours -- but a fork commit per shim
makes swapping them in and out a rebuild, and the point right now is to listen.

Every active shim is recorded in the take's id and its provenance, because a
patched run that does not say so is a take nobody can place later.

Nothing here imports from ComfyUI. Where the two stacks compute an op
differently, the difference is reimplemented from their source rather than
borrowed from their process. A borrowed op makes a row that cannot be rebuilt
from this repository alone.

Difference #2 -- the fused comfy_kitchen q/k norm + rotary kernel -- is closed
and has no shim. It was measured while the kernel was still callable: with
norm_c applied, adding the kernel produced bit-identical tokens, latents and
audio. Its only content is the q/k normalisation, which norm_c already supplies
at every site. A hand-written version of its rotary half was tried and measured
further from the target, not closer, so the engine's split-half rotary is the
one that matches. There is nothing left of #2 to switch on.

One shim is one op. An earlier `qkrope` shim replaced the whole of
Attention.project_qkv with comfy_kitchen.rms_rope_split_half, which was two
changes behind one flag -- the q/k normalisation AND the rotary -- so its row in
every table was not a single-variable result. It is gone. `rmsnorm` already
carries the normalisation half wherever it occurs, and `rope` carries the rotary
half alone. Patch at the narrowest boundary that still reproduces the op, or the
measurement is not attributable to anything.

A caution that no longer holds, kept because it was quoted for weeks: an earlier
bisect recorded shims 1+2 without 3 at a mean logit difference of 0.0209 against
0.0169 unshimmed -- worse than none. That bisect ran with torch mismatched
between the stacks. With torch aligned, blockattn is inert in every combination
and nothing gets worse. Do not re-quote the 0.0209 figure.
"""
from __future__ import annotations

_ORIGINALS: dict[str, object] = {}


def _norm_patch(name, forward):
    """Install one rung of the RMSNorm ladder. The rungs are mutually exclusive."""
    from yue2 import modeling_yue2
    _ORIGINALS[name] = modeling_yue2.RMSNorm.forward
    modeling_yue2.RMSNorm.forward = forward


def _norm_revert(name):
    from yue2 import modeling_yue2
    modeling_yue2.RMSNorm.forward = _ORIGINALS.pop(name)


def _apply_norm_a():
    """One change: stop rounding the scale to bf16 before it multiplies x.

    modeling_yue2.py:132 is three roundings, not one:
        x * rsqrt(x.float().pow(2).mean(-1) + eps).to(x.dtype) * self.weight
                                              R1 ^^^^^^^^^^^^  R2         R3
    R1 rounds the fp32 reciprocal square root to bf16. R2 rounds the product
    with x. R3 rounds the product with the weight. This rung removes R1 alone:
    the scale stays in fp32 and the normalised value is rounded once, which is
    R2 landing in a different place rather than an extra rounding removed.
    The weight still multiplies in bf16 afterwards, exactly as before.
    """
    def forward(self, x):
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale).to(x.dtype) * self.weight

    _norm_patch("norm_a", forward)


def _revert_norm_a():
    _norm_revert("norm_a")


def _apply_norm_b():
    """norm_a plus one change: apply the weight in fp32, round once at the end.

    Removes R3. Everything up to the output is now fp32 and a single rounding
    produces the bf16 result, which is what a fused kernel does. If this rung
    and norm_c agree, F.rms_norm is exactly these three roundings collapsed and
    nothing else; if they disagree, the op does a fourth thing worth finding.
    """
    def forward(self, x):
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale * self.weight.float()).to(x.dtype)

    _norm_patch("norm_b", forward)


def _revert_norm_b():
    _norm_revert("norm_b")


def _apply_norm_c():
    """The op itself: torch.nn.functional.rms_norm, which is what ComfyUI calls.

    comfy/ops.py:687 calls F.rms_norm(input, shape, weight, eps). This rung is
    the real substitution rather than an imitation of it -- the ladder above
    exists to say which of its differences account for the result, and this rung
    exists so the answer is measured against the actual op and not against my
    reconstruction of it.
    """
    import torch

    def forward(self, x):
        return torch.nn.functional.rms_norm(x, (self.weight.shape[-1],), self.weight, self.eps)

    _norm_patch("norm_c", forward)


def _revert_norm_c():
    _norm_revert("norm_c")


def _apply_norm_metal():
    """RMSNorm using the actual AppleSilicon-FP8 reduction kernel, locally vendored."""
    from .vendor.metal_norm import fused_rmsnorm_modulate

    def forward(self, x):
        d = self.weight.numel()
        return fused_rmsnorm_modulate(x.contiguous().view(-1, d),
                                     self.weight.contiguous().view(d), self.eps).view_as(x)

    _norm_patch("norm_metal", forward)


def _revert_norm_metal():
    _norm_revert("norm_metal")


def _apply_qkrope():
    """2: the fused q/k norm + split-half rotary, transcribed from the eager backend.

    comfy_kitchen ships a pure-PyTorch implementation alongside its compiled
    kernels, in backends/eager/rope.py, and that is what this is copied from --
    no import, and no guessing at what the op does. _rms_rope1 is:

        x_norm = F.rms_norm(x, (x.shape[-1],), weight=scale, eps=epsilon)
        return apply_rope_split_half1(x_norm, freqs_cis)

    and apply_rope_split_half1 pairs element i with i + head_dim//2, multiplies
    by the two columns of the rotation and sums.

    The important part is what it does NOT do. It calls F.rms_norm and lets the
    result land in bf16, then rotates that bf16 value -- the same rounding the
    engine has. "Fused" here means one call, not one rounding. A version that
    carried fp32 through both halves was written from the docstring and measured
    4.27% against this path's 18.04%, which is how a plausible reconstruction
    gets caught.

    Compound by construction: the normalisation half is exactly norm_c, so this
    shim run alone applies F.rms_norm to q and k and nothing else in the network.
    Its single-variable reading is the increment on top of norm_c.
    """
    import torch
    import torch.nn.functional as F
    from yue2 import modeling_yue2

    _ORIGINALS["qkrope"] = modeling_yue2.Attention.project_qkv

    def norm_rope(x, weight, eps, freqs_cis):
        x_norm = F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)
        t = x_norm.reshape(*x_norm.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        t = t.to(freqs_cis.dtype)
        out = freqs_cis[..., 0] * t[..., 0] + freqs_cis[..., 1] * t[..., 1]
        return out.movedim(-1, -2).reshape(*x_norm.shape).type_as(x_norm)

    def project_qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        # model.py:58, verbatim in shape and order.
        rotation = torch.stack((cos, -sin, sin, cos), dim=-1)
        rotation = rotation.reshape(*cos.shape, 2, 2).unsqueeze(2).to(q.dtype)
        q = norm_rope(q, self.q_norm.weight, self.q_norm.eps, rotation)
        k = norm_rope(k, self.k_norm.weight, self.k_norm.eps, rotation)
        return q, k, v

    modeling_yue2.Attention.project_qkv = project_qkv


def _revert_qkrope():
    from yue2 import modeling_yue2
    modeling_yue2.Attention.project_qkv = _ORIGINALS.pop("qkrope")


def _apply_attn_a():
    """3a, one change: an explicit -inf float mask in place of is_causal=True.

    modeling_yue2.py:212 issues sdpa(q, k, v, is_causal=(T > 1 and k.shape[2] == T)).
    model.py:23-27 never uses is_causal; it always builds a float mask, zeros
    where a key is visible and -inf where it is not, and passes it as attn_mask.

    An explicit mask and is_causal=True describe the same attention, but they are
    not the same code path inside SDPA: the flag lets the kernel skip masked
    blocks entirely, while a float mask is added to the scores before the softmax.
    The sequence is not blocked here -- one mask over the whole prefill. Blocking
    is 3b, and it stacks on this.
    """
    import torch
    from yue2 import modeling_yue2

    _ORIGINALS["attn_a"] = modeling_yue2.Attention.forward

    def forward(self, x, cos, sin, past_key_value=None, layer_idx=0,
                attention_mask=None, cache_position=None):
        B, T, _ = x.shape
        q, k, v = self.project_qkv(x, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, layer_idx, {"cache_position": cache_position})
        if attention_mask is not None:
            out = modeling_yue2.sdpa(q, k, v, attn_mask=attention_mask[..., :k.shape[2]])
        elif not (T > 1 and k.shape[2] == T):
            out = modeling_yue2.sdpa(q, k, v, is_causal=False)
        else:
            rows = torch.arange(T, device=q.device)[:, None]
            cols = torch.arange(T, device=q.device)[None]
            mask = torch.zeros((T, T), dtype=q.dtype, device=q.device)
            mask.masked_fill_(cols > rows, -torch.inf)
            out = modeling_yue2.sdpa(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    modeling_yue2.Attention.forward = forward


def _revert_attn_a():
    from yue2 import modeling_yue2
    modeling_yue2.Attention.forward = _ORIGINALS.pop("attn_a")


def _apply_attn_b():
    """3b, one change on top of 3a: split the causal prefill into 256-query blocks.

    model.py:18-19 walks the queries in blocks of `256 if causal else length`,
    and each block attends over keys [0, block_end) only. The engine issues one
    SDPA over the whole sequence. Same attention either way; the softmax
    normalisation runs over a different set of accumulations.

    The mask form is 3a's and is unchanged here, so this rung isolates the
    blocking alone. The full ComfyUI path -- mask rank, reshape order and their
    GQA policy -- is 3c.
    """
    import torch
    from yue2 import modeling_yue2

    _ORIGINALS["attn_b"] = modeling_yue2.Attention.forward

    def forward(self, x, cos, sin, past_key_value=None, layer_idx=0,
                attention_mask=None, cache_position=None):
        B, T, _ = x.shape
        q, k, v = self.project_qkv(x, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, layer_idx, {"cache_position": cache_position})
        if attention_mask is not None:
            out = modeling_yue2.sdpa(q, k, v, attn_mask=attention_mask[..., :k.shape[2]])
        elif not (T > 1 and k.shape[2] == T):
            out = modeling_yue2.sdpa(q, k, v, is_causal=False)
        else:
            blocks = []
            for start in range(0, T, 256):
                end = min(start + 256, T)
                rows = torch.arange(start, end, device=q.device)[:, None]
                cols = torch.arange(end, device=q.device)[None]
                mask = torch.zeros((end - start, end), dtype=q.dtype, device=q.device)
                mask.masked_fill_(cols > rows, -torch.inf)
                blocks.append(modeling_yue2.sdpa(q[:, :, start:end], k[:, :, :end],
                                                 v[:, :, :end], attn_mask=mask))
            out = torch.cat(blocks, dim=2)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    modeling_yue2.Attention.forward = forward


def _revert_attn_b():
    from yue2 import modeling_yue2
    modeling_yue2.Attention.forward = _ORIGINALS.pop("attn_b")


def _apply_gqa():
    """4, one change: who expands the KV heads when there is no attention mask.

    comfy/ops.py:56-61 expands k and v for grouped-query attention only when an
    attn_mask is present, and otherwise hands enable_gqa=True to torch. The pack
    passes a mask on prefill and None on every decode step (model.py:22), so
    ComfyUI runs native GQA for 1123 of the riff's 1125 tokens.

    modeling_yue2.py:23-28 expands unconditionally on MPS, because enable_gqa is
    not implemented on every MPS release. Same maths, different kernel, and on
    the decode path -- which is nearly the whole take.
    """
    import torch
    import torch.nn.functional as F
    from yue2 import modeling_yue2

    _ORIGINALS["gqa"] = modeling_yue2.sdpa

    def sdpa(query, key, value, *, attn_mask=None, is_causal=False):
        grouped = query.shape[1] != key.shape[1]
        if grouped and attn_mask is not None:
            groups = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
            grouped = False
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=grouped)

    modeling_yue2.sdpa = sdpa


def _revert_gqa():
    from yue2 import modeling_yue2
    modeling_yue2.sdpa = _ORIGINALS.pop("gqa")


def _apply_lastnorm():
    """8, one change: apply the final norm AFTER slicing, the way ComfyUI does.

    modeling_yue2.py:443 ends Backbone.forward with

        return self.norm(x), past_key_values

    so the final RMSNorm runs over the whole sequence -- 2149 rows on prefill --
    and YuE2ForCausalLM.forward:554 slices the last row out of the already
    normalised tensor. model.py:156 is the other way round:

        logits = self.lm_head(self.model.norm(x[:, -logits_to_keep:]))

    one row is sliced first, and only that row is normalised.

    The two agree only if the norm is row-independent, and the engine's is not.
    torch.mean on MPS reduces differently depending on the tensor's shape, so
    normalising row 2148 inside a 2149-row tensor is not bit-equal to
    normalising it alone -- measured at 21 of 2149 rows differing at
    hidden_size. F.rms_norm IS row-independent, per-row equal to whole, which
    predicts this rung is live without norm_c and inert with it.

    Only the logits see the change; the KV cache and every residual are
    untouched, because the final norm feeds nothing else.
    """
    import torch
    from yue2 import modeling_yue2

    lm = modeling_yue2.YuE2ForCausalLM
    _ORIGINALS["lastnorm"] = lm.forward
    original = lm.forward

    class NormThenHead(torch.nn.Module):
        def __init__(self, norm, head):
            super().__init__()
            self.norm, self.head = norm, head

        def forward(self, x):
            return self.head(self.norm(x))

    def forward(self, *args, **kwargs):
        norm, head = self.model.norm, self.lm_head
        self.model.norm = torch.nn.Identity()
        self.lm_head = NormThenHead(norm, head)
        try:
            return original(self, *args, **kwargs)
        finally:
            self.model.norm, self.lm_head = norm, head

    lm.forward = forward


def _revert_lastnorm():
    from yue2 import modeling_yue2
    modeling_yue2.YuE2ForCausalLM.forward = _ORIGINALS.pop("lastnorm")


def _apply_cache():
    """9, one change: store the KV cache in ComfyUI's layout, not the engine's.

    modeling_yue2.py:371-378 allocates [B, num_kv_heads, T, D] with torch.zeros
    and writes [:, :, pos:end] = k, returning a contiguous [B, H, :end, D] view.
    model.py:119-131 allocates [B, T, H, D] with torch.empty and writes
    [:, pos:end].copy_(k), returning [B, :end, H, D], which model.py:16 then
    transposes into [B, H, T, D] -- a NON-contiguous view whose stride over T is
    H*D rather than D.

    The values are the same. The strides are not, and strides are what SDPA
    dispatches on: a contiguous [B,H,T,D] and a transposed [B,T,H,D] are
    different inputs to the kernel even though they index the same numbers. This
    is the one structural difference left in the AR path after everything else
    was applied or eliminated, and it touches every attention call at every
    layer and every step.

    The replacement accepts and returns the engine's [B, H, T, D] convention so
    nothing else has to change; it transposes at the boundary, which is exactly
    what makes the tensor handed to SDPA carry ComfyUI's strides.
    """
    import torch
    from yue2 import modeling_yue2

    _ORIGINALS["cache"] = modeling_yue2.StaticKVCache

    class PackLayoutCache:
        def __init__(self, num_layers, batch_size, num_kv_heads, max_seq_len,
                     head_dim, dtype, device):
            self.num_layers = num_layers
            self.max_seq_len = max_seq_len
            self.position = 0
            self.keys = [torch.empty(batch_size, max_seq_len, num_kv_heads, head_dim,
                                     dtype=dtype, device=device) for _ in range(num_layers)]
            self.values = [torch.empty_like(t) for t in self.keys]

        def get_seq_length(self, layer_idx=0):
            return self.position

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            end = self.position + k.shape[1]
            if end > self.max_seq_len:
                raise ValueError(f"KV cache capacity {self.max_seq_len} exceeded by {end}")
            self.keys[layer_idx][:, self.position:end].copy_(k)
            self.values[layer_idx][:, self.position:end].copy_(v)
            if layer_idx == self.num_layers - 1:
                self.position = end
            return (self.keys[layer_idx][:, :end].transpose(1, 2),
                    self.values[layer_idx][:, :end].transpose(1, 2))

        def reset(self):
            self.position = 0

    modeling_yue2.StaticKVCache = PackLayoutCache


def _revert_cache():
    from yue2 import modeling_yue2
    modeling_yue2.StaticKVCache = _ORIGINALS.pop("cache")


def _apply_nograd():
    """6, one change: run the sampling loop outside torch.inference_mode.

    sampling.py:73 decorates generate_tokens with @torch.inference_mode(); the
    pack's equivalent has no decorator. Inference mode is not just no-grad -- it
    marks tensors so they cannot be used in autograd later, and some kernels
    dispatch differently under it.

    pipeline.py:16 binds the name at import (`from .sampling import
    generate_tokens`), so patching only sampling would leave the caller on the
    decorated original. Both names are replaced.
    """
    from yue2 import pipeline, sampling

    _ORIGINALS["nograd"] = (sampling.generate_tokens, pipeline.generate_tokens)
    raw = sampling.generate_tokens.__wrapped__
    sampling.generate_tokens = raw
    pipeline.generate_tokens = raw


def _revert_nograd():
    from yue2 import pipeline, sampling
    sampling.generate_tokens, pipeline.generate_tokens = _ORIGINALS.pop("nograd")


def _apply_alllogits():
    """7, one change: drop logits_to_keep on the single-token decode calls.

    sampling.py:156 passes logits_to_keep=1; the pack (sampling.py:125) omits it
    and takes the engine default, which is 0 -- meaning "keep them all"
    (modeling_yue2.py:554). With one token in flight both select the same row,
    so this is a slicing difference rather than an arithmetic one:
    hidden_states[:, -1:, :] versus hidden_states itself, and the lm_head matmul
    then runs on a narrowed view in one case and the whole tensor in the other.

    Prefill still passes logits_to_keep=1 in both stacks, so only the decode
    calls are touched -- the shim keys off a single-position input.
    """
    from yue2 import modeling_yue2

    cls = modeling_yue2.YuE2ForCausalLM
    _ORIGINALS["alllogits"] = cls.forward
    original = cls.forward

    def forward(self, input_ids=None, *args, **kwargs):
        if input_ids is not None and input_ids.shape[-1] == 1:
            kwargs["logits_to_keep"] = 0
        return original(self, input_ids, *args, **kwargs)

    cls.forward = forward


def _revert_alllogits():
    from yue2 import modeling_yue2
    modeling_yue2.YuE2ForCausalLM.forward = _ORIGINALS.pop("alllogits")


def _apply_attn_c():
    """3c, the endpoint: ComfyUI's whole AR attention path, reimplemented here.

    Written from their source, importing nothing from ComfyUI. The chain being
    reproduced is model.py:12-28 -> attention_pytorch (attention.py:545) ->
    comfy.ops.scaled_dot_product_attention (ops.py:56). Under
    --use-pytorch-cross-attention on a non-CPU device, optimized_attention and
    optimized_attention_masked are both attention_pytorch (attention.py:870,887),
    so the dispatcher collapses to one function and there is nothing to import.

    What this adds over 3b, which was my own arrangement of the same idea:

      * the mask is rank 4. attention_pytorch:554-560 unsqueezes a 2D mask to
        [1, 1, Tq, Tk] before SDPA. A 2D mask broadcasts to the same numbers but
        is not necessarily the same kernel selection, and 3b passed 2D.
      * the per-block output is transposed and flattened to [B, Tblock, H*D]
        before the blocks are concatenated on dim 1, then reshaped back to
        [B, T, H, D] (model.py:28) and flattened again by the caller
        (model.py:154). 3b concatenated [B, H, T, D] on dim 2 and reshaped once.
        Same values, different intermediate layouts, and layout is what decides
        which SDPA path runs.
      * GQA follows their policy rather than the engine's: expand k/v only when a
        mask is present (ops.py:58-60, repeat_interleave, identical to the
        engine's), and otherwise hand enable_gqa through to torch. That makes
        difference #4 a subset of this rung -- stated, not hidden.

    Blocking at 256 and the -inf mask form come from 3a and 3b unchanged, so this
    is cumulative on them: one rung, adding mask rank, reshape order and the GQA
    policy at once because they are one function's behaviour, not three knobs.

    The attention_mask branch is left on the engine's path: ComfyUI's AR model
    has no such parameter, so there is nothing of theirs to reproduce there.
    """
    import torch
    import torch.nn.functional as F
    from yue2 import modeling_yue2

    _ORIGINALS["attn_c"] = modeling_yue2.Attention.forward

    def forward(self, x, cos, sin, past_key_value=None, layer_idx=0,
                attention_mask=None, cache_position=None):
        B, T, _ = x.shape
        q, k, v = self.project_qkv(x, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, layer_idx, {"cache_position": cache_position})
        if attention_mask is not None:
            out = modeling_yue2.sdpa(q, k, v, attn_mask=attention_mask[..., :k.shape[2]])
            return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

        heads, head_dim = q.shape[1], q.shape[3]
        causal = T > 1 and k.shape[2] == T
        block = 256 if causal else T
        outputs = []
        for start in range(0, T, block):
            end = min(start + block, T)
            key_end = end if causal else k.shape[2]
            mask = None
            if causal:
                visible = (torch.arange(key_end, device=q.device)[None]
                           <= torch.arange(start, end, device=q.device)[:, None])
                mask = torch.zeros((end - start, key_end), dtype=q.dtype, device=q.device)
                mask.masked_fill_(~visible, -torch.inf)
                mask = mask.unsqueeze(0).unsqueeze(0)
            kb, vb = k[:, :, :key_end], v[:, :, :key_end]
            grouped = heads != kb.shape[1]
            if grouped and mask is not None:
                n_rep = heads // kb.shape[1]
                kb = kb.repeat_interleave(n_rep, dim=-3)
                vb = vb.repeat_interleave(n_rep, dim=-3)
                grouped = False
            chunk = F.scaled_dot_product_attention(
                q[:, :, start:end], kb, vb, attn_mask=mask, dropout_p=0.0,
                is_causal=False, enable_gqa=grouped)
            outputs.append(chunk.transpose(1, 2).reshape(B, -1, heads * head_dim))
        out = torch.cat(outputs, dim=1).reshape(B, T, heads, head_dim)
        return self.o_proj(out.flatten(2))

    modeling_yue2.Attention.forward = forward


def _revert_attn_c():
    from yue2 import modeling_yue2
    modeling_yue2.Attention.forward = _ORIGINALS.pop("attn_c")



# Declaration order is execution order in the network, and apply() follows it.
def _apply_mtl_a():
    """11a, one change: attention runs the mtlflashattn Metal kernel, not stock MPS SDPA.

    This op is in neither yue2 tree. ComfyUI-AppleSilicon-FP8's __init__.py:98
    installs _patches/flash_attn_mtl.py at every ComfyUI startup -- unconditional,
    no launch flag -- which calls metal_flash_attn.sdpa.install() and rebinds
    torch.nn.functional.scaled_dot_product_attention (sdpa.py:176-178) to a gated
    dispatcher. Both stacks call F.scaled_dot_product_attention; in ComfyUI's
    process that name does not mean torch's op.

    The gate (metal_flash_attn/sdpa.py:33-73) fires here on the fast-tier rule:
    no mask, no dropout, rank 4, bf16, head_dim 128, Hq % Hkv == 0, and
    max(Lq, Lk) >= MTLFLASHATTN_SDPA_FAST_MIN_SEQ (1024) with a tier in
    {v2, v2_fp32, v2_bf16}. The riff's decode steps are 1 query against 2149-3274
    keys, tier v2_bf16 -- eligible from the very first step, since the gate reads
    max(Lq, Lk) and not Lq.

    This rung installs the real patch, unmodified, so every eligible call site in
    the engine routes to the kernel. That includes our prefill, which ComfyUI's
    does NOT reach: the pack builds an explicit -inf mask (model.py:24), and a
    mask disqualifies the call at sdpa.py:37-38. Rung b narrows to what ComfyUI
    actually dispatches.
    """
    from metal_flash_attn import sdpa as mfa_sdpa
    _ORIGINALS["mtl_a"] = mfa_sdpa.install()


def _revert_mtl_a():
    from metal_flash_attn import sdpa as mfa_sdpa
    if _ORIGINALS.pop("mtl_a"):
        mfa_sdpa.uninstall()


def _apply_mtl_b():
    """11b, the same op as 11a, narrowed to the calls ComfyUI actually routes.

    ComfyUI's prefill carries a mask and stays on stock SDPA; its decode steps
    pass mask=None and reach the kernel. The engine's prefill passes no mask and
    leans on is_causal (modeling_yue2.py:213), so installing the patch globally
    would route a call ComfyUI leaves alone. This rung keeps the kernel for the
    unmasked, non-causal decode calls and leaves everything else on torch.

    Measured first: the kernel returns bit-identical output for 16/8 GQA and for
    16/16 pre-expanded k/v at these shapes, so this rung does not smuggle in #4
    -- whether the engine expands the KV heads makes no difference once the call
    lands on the kernel.
    """
    import math
    import os
    import torch.nn.functional as F
    from metal_flash_attn import sdpa as mfa_sdpa
    from metal_flash_attn._kernel import flash_attn_forward
    from metal_flash_attn.sdpa import _eligibility
    from yue2 import modeling_yue2

    # _eligibility reads module globals that only install() populates, and we are
    # deliberately not installing the global patch. Set them the way install()
    # does (sdpa.py:167-175) -- same env vars, same defaults -- so the gate this
    # rung consults is the one ComfyUI's process consults.
    mfa_sdpa._min_score_bytes = int(float(os.environ.get("MTLFLASHATTN_SDPA_MIN_GB", "12")) * 1024 ** 3)
    mfa_sdpa._min_seq = int(os.environ.get("MTLFLASHATTN_SDPA_MIN_SEQ", "4096"))
    mfa_sdpa._fast_min_seq = int(os.environ.get("MTLFLASHATTN_SDPA_FAST_MIN_SEQ", "1024"))

    _ORIGINALS["mtl_b"] = modeling_yue2.sdpa

    def sdpa(query, key, value, *, attn_mask=None, is_causal=False):
        grouped = query.shape[1] != key.shape[1]
        if attn_mask is None and not is_causal:
            eligible, _ = _eligibility(query, key, value, None, 0.0, False)
            if eligible:
                return flash_attn_forward(query, key, value,
                                          scale=1.0 / math.sqrt(query.shape[-1]), causal=False)
        if grouped and query.device.type == "mps":
            groups = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
            grouped = False
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=grouped)

    modeling_yue2.sdpa = sdpa


def _revert_mtl_b():
    from yue2 import modeling_yue2
    modeling_yue2.sdpa = _ORIGINALS.pop("mtl_b")


def _apply_nar_mtl():
    """Change only NAR's SDPA dispatch, preserving its existing query tiling."""
    import math
    import os
    from yue2 import nar
    from metal_flash_attn import sdpa as mfa
    from metal_flash_attn._kernel import flash_attn_forward

    original = nar.F
    _ORIGINALS["nar_mtl"] = original
    mfa._min_score_bytes = int(float(os.environ.get("MTLFLASHATTN_SDPA_MIN_GB", "12")) * 1024 ** 3)
    mfa._min_seq = int(os.environ.get("MTLFLASHATTN_SDPA_MIN_SEQ", "4096"))
    mfa._fast_min_seq = int(os.environ.get("MTLFLASHATTN_SDPA_FAST_MIN_SEQ", "1024"))

    class Functional:
        def __getattr__(self, name):
            return getattr(original, name)

        def scaled_dot_product_attention(self, q, k, v, attn_mask=None,
                                         dropout_p=0.0, is_causal=False, scale=None, **kwargs):
            eligible, _ = mfa._eligibility(q, k, v, attn_mask, dropout_p, is_causal)
            if eligible:
                return flash_attn_forward(q, k, v,
                                          scale=scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]),
                                          causal=is_causal)
            return original.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                        dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

    nar.F = Functional()


def _revert_nar_mtl():
    from yue2 import nar
    nar.F = _ORIGINALS.pop("nar_mtl")


SHIMS = {
    "norm_a": (_apply_norm_a, _revert_norm_a,
               "RMSNorm: scale kept in fp32 instead of rounded to bf16 first"),
    "norm_b": (_apply_norm_b, _revert_norm_b,
               "RMSNorm: norm_a plus the weight applied in fp32, one rounding at the end"),
    "norm_c": (_apply_norm_c, _revert_norm_c,
               "RMSNorm: torch.nn.functional.rms_norm, the op ComfyUI calls"),
    "norm_metal": (_apply_norm_metal, _revert_norm_metal,
                   "RMSNorm: standalone copy of the runtime's Metal reduction kernel"),
    "qkrope": (_apply_qkrope, _revert_qkrope,
               "2: fused q/k norm + split-half rotary, reimplemented (compound)"),
    "attn_a": (_apply_attn_a, _revert_attn_a,
               "3a: explicit -inf float mask over the whole prefill instead of is_causal"),
    "attn_b": (_apply_attn_b, _revert_attn_b,
               "3b: attn_a plus causal prefill split into 256-query blocks"),
    "attn_c": (_apply_attn_c, _revert_attn_c,
               "3c: ComfyUI's whole AR attention path, reimplemented (includes #4)"),
    "gqa": (_apply_gqa, _revert_gqa,
            "4: native enable_gqa on decode instead of expanding the KV heads"),
    "lastnorm": (_apply_lastnorm, _revert_lastnorm,
                 "8: final norm applied after slicing to the kept rows, not before"),
    "cache": (_apply_cache, _revert_cache,
              "9: KV cache in ComfyUI's [B,T,H,D] layout, transposed into SDPA"),
    "nograd": (_apply_nograd, _revert_nograd,
               "6: sampling loop outside torch.inference_mode"),
    "alllogits": (_apply_alllogits, _revert_alllogits,
                  "7: no logits_to_keep on the single-token decode calls"),
    "mtl_a": (_apply_mtl_a, _revert_mtl_a,
              "11a: mtlflashattn's real SDPA patch installed globally, as ComfyUI's startup does"),
    "mtl_b": (_apply_mtl_b, _revert_mtl_b,
              "11b: the Metal kernel only on the unmasked decode calls, as ComfyUI actually dispatches"),
    "nar_mtl": (_apply_nar_mtl, _revert_nar_mtl,
                "NAR SDPA uses the reference Metal kernel dispatch, keeping existing query tiling"),
}


EXCLUSIVE = ({"norm_a", "norm_b", "norm_c", "norm_metal"}, {"attn_a", "attn_b", "attn_c"},
             {"gqa", "mtl_a", "mtl_b"})


def apply(names) -> list[str]:
    """Switch on each named shim. Returns them in execution order."""
    unknown = [n for n in names if n not in SHIMS]
    if unknown:
        raise ValueError(f"unknown shim {unknown}; known: {sorted(SHIMS)}")
    for ladder in EXCLUSIVE:
        rungs = [n for n in names if n in ladder]
        if len(rungs) > 1:
            raise ValueError(f"{rungs} are rungs of one ladder; they patch the same method, pick one")
    ordered = [n for n in SHIMS if n in set(names)]
    for name in ordered:
        if name not in _ORIGINALS:
            SHIMS[name][0]()
    return ordered


def revert_all() -> None:
    for name in list(_ORIGINALS):
        SHIMS[name][1]()


def active() -> list[str]:
    return [n for n in SHIMS if n in _ORIGINALS]

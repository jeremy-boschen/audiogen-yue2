"""Attach LoRA adapters to the engine's model through its on_model_ready hook.

The engine constructs its model lazily, sets it to None on close(), moves it to
CPU around decoding, prepares it for fp8 on the AR path and restores it before
NAR. Anything that patches the model from outside therefore has to re-apply when
the engine hands it over, which is what on_model_ready is for: it fires on every
stage entry, not just construction, and says which stage is about to run.

Two tensor shapes appear in these files:

* a low-rank pair, ``<module>.lora_down.weight`` [r,in] and
  ``<module>.lora_up.weight`` [out,r], contributing ``up @ down``;
* a dense ``<module>.diff`` (and optional ``.diff_b``), a full-size delta used
  where a projection is small enough that low rank buys nothing -- llm2vae and
  vae2llm are 64-wide.

Both are *deltas*: applying at strength 0 must reproduce the base model exactly,
which is the property worth testing, because it fails loudly if the mapping is
wrong.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

# Which branch each module belongs to. llm2vae and vae2llm look shared by name but
# are not: every use of them in the engine is on the NAR path -- nar.py, and the
# NAR section of the model's forward. Treating them as "either" silently let a NAR
# adapter apply during AR.
AR_MODULES = frozenset({"self_attn", "mlp"})
NAR_MODULES = frozenset({"nar_self_attn", "nar_mlp", "llm2vae", "vae2llm"})


@dataclass
class Adapter:
    path: Path
    branch: str                  # "ar" or "nar"
    strength: float = 1.0

    def __post_init__(self):
        self.path = Path(self.path).expanduser()
        if self.branch not in {"ar", "nar"}:
            raise ValueError(f"lora branch must be 'ar' or 'nar', not {self.branch!r}")
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        present = branches_in(self.path)
        if present and self.branch not in present:
            # deltas() catches this too, but only once the model is in memory.
            # The file's own key names answer it from the header, so a mislabelled
            # adapter fails during --dry-run instead of 14 minutes later.
            # Only when the file *does* classify: an unclassifiable one has more
            # precise errors waiting in deltas(), and they should be the ones seen.
            raise ValueError(f"{self.path.name} carries {'/'.join(sorted(present))} modules, "
                             f"so branch {self.branch!r} would apply nothing")

    def identity(self) -> dict:
        """Recorded per take, so a rendered song names the adapter that shaped it."""
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        return {"path": str(self.path), "branch": self.branch,
                "strength": self.strength, "sha256_16": digest}


def branches_in(path: Path) -> set[str]:
    """Which branches an adapter file actually touches, from its header alone."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
    return {b for b in (branch_of(k) for k in keys) if b is not None}


def read(path: Path) -> tuple[dict, dict]:
    from safetensors.torch import load_file
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    return load_file(str(path)), metadata


def _targets(tensors: dict) -> dict:
    """Group flat adapter keys by the module each one modifies."""
    grouped: dict[str, dict] = {}
    for key, value in tensors.items():
        for suffix, slot in ((".lora_down.weight", "down"), (".lora_up.weight", "up"),
                             (".diff_b", "diff_b"), (".diff", "diff")):
            if key.endswith(suffix):
                grouped.setdefault(key[: -len(suffix)], {})[slot] = value
                break
        else:
            raise ValueError(f"unrecognised LoRA key {key!r}")
    return grouped


def branch_of(module_name: str) -> str | None:
    """Which branch a module sits on, or None if it cannot be classified."""
    for part in module_name.split("."):
        if part in NAR_MODULES:
            return "nar"
        if part in AR_MODULES:
            return "ar"
    return None


def _resolve(model, dotted: str):
    target = model
    for part in dotted.split("."):
        if not hasattr(target, part):
            return None
        target = getattr(target, part)
    return target


def deltas(model, adapter: Adapter) -> dict[str, torch.Tensor]:
    """The weight delta each module would receive, without applying anything."""
    tensors, _ = read(adapter.path)
    out: dict[str, torch.Tensor] = {}
    for name, parts in _targets(tensors).items():
        module = _resolve(model, name)
        if module is None or not hasattr(module, "weight"):
            raise KeyError(f"adapter targets {name!r}, which this model does not have")
        where = branch_of(name)
        if where is None:
            # Better to stop than to guess: applying a delta on the wrong stage is
            # silent, and shows up only as a take that sounds subtly off.
            raise ValueError(f"cannot tell which branch {name!r} belongs to; "
                             "add it to AR_MODULES or NAR_MODULES")
        if where != adapter.branch:
            continue
        if "diff" in parts:
            delta = parts["diff"]
        elif "down" in parts and "up" in parts:
            delta = parts["up"].to(torch.float32) @ parts["down"].to(torch.float32)
        else:
            raise ValueError(f"{name!r} has neither a diff nor a complete lora pair")
        if tuple(delta.shape) != tuple(module.weight.shape):
            raise ValueError(f"{name!r}: adapter delta {tuple(delta.shape)} does not fit "
                             f"weight {tuple(module.weight.shape)}")
        out[name] = delta
        if "diff_b" in parts:
            out[name + ".__bias__"] = parts["diff_b"]
    if not out:
        raise ValueError(f"{adapter.path.name} contributed nothing on branch {adapter.branch!r}")
    return out


def hook(adapters: list[Adapter]):
    """Build an on_model_ready callback that applies `adapters` to the right stage.

    The callback must be idempotent: the engine fires it on every stage entry, so
    a naive additive apply would compound. Each modified tensor's base value is
    kept on the model the first time it is touched, and every apply starts from
    that base rather than from whatever the last one left behind.
    """
    def apply(model, *, for_nar, fresh, loaded):
        stage = "nar" if for_nar else "ar"
        base = getattr(model, "_audiogen_lora_base", None)
        if base is None or fresh:
            base = {}
            model._audiogen_lora_base = base

        wanted = [a for a in adapters if a.branch == stage]
        for name in list(base):
            module = _resolve(model, name.removesuffix(".__bias__"))
            tensor = module.bias if name.endswith(".__bias__") else module.weight
            with torch.no_grad():
                tensor.copy_(base[name].to(tensor.device, tensor.dtype))

        for adapter in wanted:
            for name, delta in deltas(model, adapter).items():
                is_bias = name.endswith(".__bias__")
                module = _resolve(model, name.removesuffix(".__bias__"))
                tensor = module.bias if is_bias else module.weight
                if name not in base:
                    base[name] = tensor.detach().clone().cpu()
                with torch.no_grad():
                    tensor.add_(delta.to(tensor.device, torch.float32).to(tensor.dtype),
                                alpha=adapter.strength)
    return apply


def from_manifest(entries) -> list[Adapter]:
    return [Adapter(path=e["path"], branch=e["branch"], strength=e.get("strength", 1.0))
            for e in (entries or [])]

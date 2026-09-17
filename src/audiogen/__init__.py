"""Conventions layered over a pinned YuE2 inference fork.

The split this package exists to keep:

* The **fork** (jeremy-boschen/YuE, pinned in ``env/pins.env``) holds changes to
  what the engine computes -- expressed in the library's own vocabulary, inert by
  default, and shaped so they could go upstream. Five commits today: an MPS stage
  guard, ``carry=`` continuation, acoustic chunk pinning, an ``on_model_ready``
  hook, and ``rng_device``.
* **This package** holds everything specific to us: LoRA loading through that
  hook, continuation chains, seeds and naming, provenance records, delivery.

Anything that alters engine numerics belongs in the fork, because every commit
there is a rebase cost forever and mixing conventions in is how a fork stops
tracking upstream. Anything that is merely how *we* drive the engine belongs here.
"""

__version__ = "0.1.0"

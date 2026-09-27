"""Stable digests for stage artifacts.

Hashing is how every claim in this repo is checked, so the rules are fixed here
rather than re-invented per script:

* integer token streams are hashed as int64 regardless of how they arrived, so a
  list and an int32 array of the same tokens agree;
* float arrays include dtype and shape in the digest, because an array that
  reshaped or changed precision is a different artifact even if the bytes line up;
* 16 hex characters is enough to compare by eye in a terminal and in a commit
  message, which is where these actually get read.
"""
import hashlib

import numpy as np

WIDTH = 16


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:WIDTH]


def hash_tokens(tokens) -> str:
    return digest(np.asarray(list(tokens), dtype=np.int64).tobytes())


def hash_array(array) -> str:
    array = np.ascontiguousarray(array)
    return digest(f"{array.dtype}|{array.shape}|".encode() + array.tobytes())

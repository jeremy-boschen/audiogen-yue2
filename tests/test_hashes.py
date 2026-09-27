"""The digest rules, since every claim in this repo rests on them."""
import numpy as np

from audiogen import hashes


def test_token_hash_ignores_container_and_int_width():
    # A take's tokens come back as a list, are stored as int32, reload as an array.
    # All three must agree or a round trip looks like a regression.
    tokens = [1, 2, 3, 70000]
    assert (hashes.hash_tokens(tokens)
            == hashes.hash_tokens(np.asarray(tokens, dtype=np.int32))
            == hashes.hash_tokens(np.asarray(tokens, dtype=np.int64)))


def test_array_hash_separates_shape_and_dtype():
    flat = np.arange(8, dtype=np.float32)
    assert hashes.hash_array(flat) != hashes.hash_array(flat.reshape(4, 2))
    assert hashes.hash_array(flat) != hashes.hash_array(flat.astype(np.float64))


def test_array_hash_is_stable_across_non_contiguous_views():
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert hashes.hash_array(base[:, :2]) == hashes.hash_array(np.ascontiguousarray(base[:, :2]))


def test_width_is_readable_in_a_terminal():
    assert len(hashes.digest(b"x")) == hashes.WIDTH == 16

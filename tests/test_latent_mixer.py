"""The latent mixer's edits: relative to each channel's own statistics, and neutral by default."""
import numpy as np
import pytest

from audiogen import latent_mixer as mixer


def take():
    rng = np.random.default_rng(0)
    return (rng.normal(size=(200, 64)) * np.linspace(0.5, 2, 64) + np.linspace(-1, 1, 64)).astype(np.float32)


def test_neutral_edits_change_nothing():
    latent = take()
    assert np.array_equal(mixer.apply(latent, mixer.neutral(), mixer.stats(latent)), latent) or \
        np.allclose(mixer.apply(latent, mixer.neutral(), mixer.stats(latent)), latent, atol=1e-6)


def test_offset_is_in_the_channels_own_standard_deviations():
    latent, edits = take(), mixer.neutral()
    whole = mixer.stats(latent)
    edits["offset"][10] = 2.0
    out = mixer.apply(latent, edits, whole)
    assert np.allclose(out[:, 10] - latent[:, 10], 2.0 * whole["std"][10], atol=1e-5)
    assert np.allclose(np.delete(out, 10, axis=1), np.delete(latent, 10, axis=1), atol=1e-6)


def test_mute_holds_the_average_and_solo_holds_everything_else():
    latent, edits = take(), mixer.neutral()
    whole = mixer.stats(latent)
    edits["mute"][3] = True
    out = mixer.apply(latent, edits, whole)
    assert np.allclose(out[:, 3], whole["mean"][3]) and np.allclose(out[:, 4], latent[:, 4], atol=1e-6)
    edits["solo"][7] = True                      # solo wins: only 7 moves, 3's mute no longer matters
    out = mixer.apply(latent, edits, whole)
    assert np.allclose(out[:, 7], latent[:, 7], atol=1e-6)
    assert np.allclose(np.delete(out, 7, axis=1), np.delete(np.broadcast_to(whole["mean"], latent.shape), 7, axis=1))


def test_gain_scales_movement_around_the_mean():
    latent, edits = take(), mixer.neutral()
    whole = mixer.stats(latent)
    edits["gain"][0] = 0.0
    edits["master_gain"] = 2.0
    out = mixer.apply(latent, edits, whole)
    assert np.allclose(out[:, 0], whole["mean"][0])
    assert np.allclose(out[:, 1] - whole["mean"][1], 2 * (latent[:, 1] - whole["mean"][1]), atol=1e-5)


def test_edits_must_cover_every_channel():
    with pytest.raises(ValueError, match="64"):
        mixer.apply(take(), {"offset": [0.0] * 3}, mixer.stats(take()))

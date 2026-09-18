"""Captured reference conversion for a small buffer (libav's scalar path)."""
import numpy as np
import soundfile as sf

from audiogen.audio import save_album_flac


def test_album_flac_preserves_reference_half_integer_conversion(tmp_path):
    scaled = np.array([[-621.5, 904.5], [-1299.5, 1008.5], [-32768, 32768]], dtype=np.float32)
    path = tmp_path / 'audio.flac'
    save_album_flac(path, scaled / 32768, 48000)
    pcm, rate = sf.read(path, dtype='int16')
    assert rate == 48000
    assert sf.info(path).subtype == 'PCM_16'
    # Small buffers use the scalar conversion path. The 45s reference also
    # exercises SIMD conversion, whose tie behavior differs on Apple Silicon.
    np.testing.assert_array_equal(pcm, [[-622, 904], [-1300, 1008], [-32768, 32767]])

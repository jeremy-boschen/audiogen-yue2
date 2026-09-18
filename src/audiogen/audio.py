"""Historical reference encoder for diagnostics; production uses the engine writer."""
from pathlib import Path

import numpy as np
from yue2.pipeline import SongResult


def save_album_flac(path, audio, sample_rate, pcm_bits=16):
    import av
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.ascontiguousarray(audio, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] not in (1, 2):
        raise ValueError('Expected mono or stereo audio shaped [frames, channels]')
    if pcm_bits not in (16, 24):
        raise ValueError('FLAC pcm_bits must be 16 or 24')
    layout = 'mono' if samples.shape[1] == 1 else 'stereo'
    # Match the album encoder, including its float-to-s16 rounding. Quantizing
    # with numpy.rint first changes half-integer samples on this runtime.
    with av.open(str(path), mode='w', format='flac') as container:
        stream = container.add_stream('flac', rate=sample_rate, layout=layout)
        if pcm_bits == 24:
            stream.format = 's32'
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format='flt', layout=layout)
        frame.sample_rate = sample_rate
        frame.pts = 0
        container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return str(path)


class AlbumSongResult(SongResult):
    def save(self, path):
        if Path(path).suffix.lower() == '.flac':
            return save_album_flac(path, self.audio, self.sample_rate,
                                   self.config.get('audiogen', {}).get('pcm_bits', 16))
        return super().save(path)

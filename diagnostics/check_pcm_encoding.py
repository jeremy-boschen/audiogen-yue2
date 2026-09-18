"""Measure the album encoder's float-to-PCM conversion, using standalone PyAV."""
import argparse,json
from pathlib import Path
import av
import numpy as np
import soundfile as sf
p=argparse.ArgumentParser();p.add_argument('audio',type=Path);p.add_argument('out',type=Path)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);x=np.load(a.audio)
with av.open(str(a.out/'reference.flac'),mode='w',format='flac') as container:
 stream=container.add_stream('flac',rate=48000,layout='stereo')
 frame=av.AudioFrame.from_ndarray(x.reshape(1,-1),format='flt',layout='stereo');frame.sample_rate=48000;frame.pts=0
 container.mux(stream.encode(frame));container.mux(stream.encode(None))
expected,sr=sf.read(a.out/'reference.flac',dtype='int16')
actual=np.rint(x*32768).clip(-32768,32767).astype(np.int16)
sf.write(a.out/'standalone.flac',actual,48000,subtype='PCM_16')
readback,_=sf.read(a.out/'standalone.flac',dtype='int16')
report=dict(reference_subtype=sf.info(a.out/'reference.flac').subtype,shape=list(expected.shape),
            quantization_exact=np.array_equal(actual,expected),file_pcm_exact=np.array_equal(readback,expected),
            unequal=int(np.count_nonzero(actual!=expected)))
(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

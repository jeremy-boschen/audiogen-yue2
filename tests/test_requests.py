"""Song request normalization and per-step text selection."""
from types import SimpleNamespace

import pytest

from audiogen.render import request_for


@pytest.mark.parametrize('abc, expected', [
    (' \nX:1\nK:C\nC4|\n', 'X:1\nK:C\nC4|'),
    ('X:1\n\nK:C\nC4|', 'X:1\n\nK:C\nC4|'),
    (' \n\t', None),
    (None, None),
])
def test_score_trims_outer_whitespace(abc, expected):
    song = SimpleNamespace(style='rock', lyrics='[verse]\nhello', id='fixture',
                           label=None, lora=[], cot='full', abc=abc, cfg_scale=None)
    step = SimpleNamespace(seed=777, id='riff')
    request = request_for(song, step)
    assert request.abc == expected
    assert request.lyrics == song.lyrics


def test_continuation_inputs_are_selected_per_step(tmp_path):
    import json
    from audiogen.song import load

    (tmp_path / 'style.txt').write_text('rock')
    (tmp_path / 'lyrics.txt').write_text('original lyrics')
    (tmp_path / 'score.abc').write_text('X:1\nK:C\nC4|\n')
    (tmp_path / 'tail.abc').write_text(' \nX:2\nK:D\nD4|\n')
    (tmp_path / 'tail.txt').write_text('new lyrics\n')
    spec = {'seed': 779, 'steps': [
        {'id': 'riff', 'seconds': 45},
        {'id': 'tail', 'seconds': 280, 'carry_from': 'riff',
         'carry_seconds': 45, 'score_file': 'tail.abc', 'lyrics_file': 'tail.txt'},
    ]}
    (tmp_path / 'song.json').write_text(json.dumps(spec))
    song = load(tmp_path)
    first = request_for(song, song.steps[0])
    tail = request_for(song, song.steps[1])
    assert first.abc == 'X:1\nK:C\nC4|'
    assert first.lyrics == 'original lyrics'
    assert tail.abc == 'X:2\nK:D\nD4|'
    assert tail.lyrics == 'new lyrics\n'
    assert tail.style == first.style == 'rock'
    (tmp_path / 'tail.abc').unlink()
    with pytest.raises(FileNotFoundError):
        load(tmp_path)

"""Extract additional parity fixtures without importing or modifying ComfyUI.

Only reads historical FLAC metadata. Reference requests are written, not queued.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess


def workflow(audio):
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format_tags=prompt',
        '-of', 'json', str(audio)], text=True))
    return json.loads(info['format']['tags']['prompt'])


def branch(graph, output):
    selected = {}

    def visit(key):
        if key in selected:
            return
        selected[key] = copy.deepcopy(graph[key])
        for value in graph[key]['inputs'].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph:
                visit(value[0])

    visit(output)
    return selected


def prepare(audio, suffix, name, seed, out):
    graph = workflow(audio)
    output = next(k for k, v in graph.items() if v['class_type'] == 'SaveAudioAdvanced'
                  and v['inputs']['filename_prefix'].endswith(suffix))
    graph = branch(graph, output)
    plan = next(v['inputs'] for v in graph.values() if v['class_type'] == 'FL_YuE2_Plan')
    render_key = next(k for k, v in graph.items() if v['class_type'] == 'FL_YuE2_Render')
    render = graph[render_key]['inputs']
    plan['seed'] = seed
    take = 'parity_goal_validation_' + name
    graph[output]['inputs']['filename_prefix'] = 'audio/YuE2/parity_goal_validation/' + name
    graph[str(max(map(int, graph)) + 1)] = {
        'class_type': 'FL_YuE2_SaveTokens',
        'inputs': {'music_tokens': [render_key, 1], 'name': take}}
    destination = out / name
    destination.mkdir(parents=True, exist_ok=False)
    for filename, key in [('style.txt', 'style'), ('lyrics.txt', 'lyrics'), ('score.abc', 'score_abc')]:
        (destination / filename).write_text(plan[key])
    spec = {'id': name, 'seed': seed, 'cot': plan['planning'], 'cfg_scale': render['guidance'],
            'generation_config': {'rng_device': 'cpu', 'ode_steps': render['acoustic_steps']},
            'semantic_sampling': {k: render[k] for k in ('temperature', 'top_p', 'top_k', 'repetition_penalty')},
            'steps': [{'id': 'riff', 'seconds': render['max_duration'],
                       'style_file': 'style.txt', 'lyrics_file': 'lyrics.txt'}]}
    (destination / 'song.json').write_text(json.dumps(spec, indent=2) + '\n')
    (destination / 'reference_request.json').write_text(json.dumps({'prompt': graph}, indent=2) + '\n')
    provenance = {'source_audio': str(audio), 'source_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
                  'reference_take': take, 'seed_override': seed,
                  'reference_audio_prefix': graph[output]['inputs']['filename_prefix']}
    (destination / 'source.json').write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-root', type=Path, default=Path.home() / 'dev/ai/ComfyUI')
    p.add_argument('--out', type=Path, default=Path('songs/parity_validation'))
    args = p.parse_args()
    archive = args.reference_root / 'output/audio/YuE2/_archive/2026-09-16_reset/02_takes_45s'
    for suffix, name, seed in [('swap_B_guitar', 'guitar_seed778', 778),
                               ('swap_B_guitar', 'guitar_seed780', 780),
                               ('swap_A_synth', 'synth_seed777', 777)]:
        prepare(archive / (suffix + '_00001.flac'), suffix, name, seed, args.out)
        print(args.out / name)


if __name__ == '__main__':
    main()

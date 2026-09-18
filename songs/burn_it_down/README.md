# Burn It Down, on the yue2 stack

Style, lyrics and seed are the album track's, carried over from the archive
(`audiogen-comfyui/songs/burn_it_down`). Seed 777 at ~200s is the length and
seed of that lineage's `b_grown` take, so this is the same material at the same
scale -- not the same audio. The archive rendered under ComfyUI on torch 2.14.0
and nothing here closes that gap (see docs/MATCHING_COMFYUI.md); treat this as
the same song played by a different performer.

Its job is to be familiar enough to judge a change by ear:

    bin/render.py burn_it_down --out <dir>/nar_off
    bin/render.py burn_it_down --out <dir>/nar_on --lora <adapter>.safetensors:nar

Then build the listening page with the engine skill's helper:

    python ~/dev/projects/YuE/skills/yue2-music/scripts/listen.py \
        <dir>/nar_off/burn_it_down/take <dir>/nar_on/burn_it_down/take \
        --output <dir>/compare

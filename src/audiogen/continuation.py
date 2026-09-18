"""Compatibility entrypoint; continuation is owned by the engine profile."""
def generate_semantic(pipe, plan, **kwargs):
    return pipe.generate_semantic(plan, **kwargs)

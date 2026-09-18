"""Compatibility import; the attributed implementation is maintained by YuE."""
from yue2.profiles.vendor import metal_norm as _implementation

def __getattr__(name):
    return getattr(_implementation, name)

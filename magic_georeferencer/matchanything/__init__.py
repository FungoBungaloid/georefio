"""
MatchAnything integration for Magic Georeferencer

The inference wrapper is imported lazily because it requires torch, which is
heavy and may not be installed until the dependency check has run.
"""

__all__ = ['MatchAnythingInference']


def __getattr__(name):
    if name == 'MatchAnythingInference':
        from .inference import MatchAnythingInference
        return MatchAnythingInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

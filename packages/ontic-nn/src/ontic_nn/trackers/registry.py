"""Lazy research-wrapper config registry."""

from .common import TrackerConfig

TRACKERS: dict[str, type[TrackerConfig]] = {}


def register_tracker(name: str):
    def register(cls: type[TrackerConfig]) -> type[TrackerConfig]:
        if name in TRACKERS and TRACKERS[name] is not cls:
            raise ValueError(f"tracker {name!r} is already registered")
        TRACKERS[name] = cls
        return cls

    return register

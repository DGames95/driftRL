"""Swappable controllers for the drift game.

All controllers share the same runtime contract (see Controller.act): they
take the RL observation plus an abstracted keyboard intent and return a
[delta, T] action.

Controllers are auto-discovered: every module in this package that defines
a Controller subclass is registered under that class's `name`. To add a
controller, drop a file in here; to remove one, delete it. Nothing else
needs editing.
"""

import importlib
import pkgutil

from .base import Controller

REGISTRY = {}
for _info in pkgutil.iter_modules(__path__):
    _mod = importlib.import_module(f".{_info.name}", __name__)
    for _obj in vars(_mod).values():
        if isinstance(_obj, type) and issubclass(_obj, Controller) and _obj is not Controller:
            REGISTRY[_obj.name] = _obj


def make_controller(name, env, **kwargs):
    """Build a controller by name (see REGISTRY), passing env and any
    controller-specific kwargs through to its constructor."""
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown controller: {name!r} (choices: {sorted(REGISTRY)})")
    return cls(env, **kwargs)


__all__ = ["Controller", "make_controller", "REGISTRY"]

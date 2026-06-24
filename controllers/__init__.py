"""Swappable controllers for the drift game.

All controllers share the same runtime contract (see Controller.act): they
take the RL observation plus an abstracted keyboard intent and return a
[delta, T] action.
"""

from .base import Controller
from .keyboard import KeyboardController
from .pid import PIDController
from .rl import RLController


def make_controller(name, env, args):
    """Build a controller by name: 'keyboard', 'rl', or 'pid'."""
    if name == "keyboard":
        return KeyboardController()
    if name == "rl":
        return RLController(args.model)
    if name == "pid":
        return PIDController(env)
    raise ValueError(f"unknown controller: {name}")


__all__ = ["Controller", "KeyboardController", "PIDController",
           "RLController", "make_controller"]

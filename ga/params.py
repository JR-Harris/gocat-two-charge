"""Frozen GA parameters (the discrete memetic GA).

The numeric operator rates live HERE, in one frozen-typed record. G
(the generation count) is deliberately NOT a parameter: the stop rule
(fixed-generations, design decision 2026-10-13) makes G an explicit
argument of every run -- `ga.ga.run_ga(..., G)` and `run_ga.py
--generations G` -- so no run can ever silently inherit a default
generation budget, and there is no evaluation-budget stop to default.
The operator rates freeze at item 6's battery pre-registration (which
also freezes G, from the statistical sweep); until then the SMOKE set
below was plumbing when it was written; the pre-declared scoping sweep
then selected exactly these values as the stage-2 frozen set, so they are
the settings of every reported battery run (the ledgers record them as
`param_set`). A
default change is one committed edit with its rationale in the coding
log.

use_descent exists for the item-6 ablation arm only (descent off);
the default run has descent ON. nn_k is asserted == 8: the descent
neighbourhood arithmetic 2 x (9 x 2 - 1) = 34 neighbours is part of
the frozen design (the standing site + its 8 nearest, two signs).
"""

import dataclasses

__all__ = ["GAParams", "PARAM_SETS", "get_param_set"]


@dataclasses.dataclass(frozen=True)
class GAParams:
    """The frozen GA operator record (see the module docstring)."""

    pop_size: int = 50
    tournament_k: int = 3
    pc: float = 0.9
    pm: float = 0.2
    nn_k: int = 8
    n_elite: int = 2
    acceptance_margin: float = 0.0
    use_descent: bool = True
    fence_rejection_cap: int = 1000

    def __post_init__(self):
        def _is_int(x):
            return isinstance(x, int) and not isinstance(x, bool)

        if not _is_int(self.pop_size) or self.pop_size < 2:
            raise ValueError(f"pop_size must be an int >= 2, got {self.pop_size!r}")
        if not _is_int(self.tournament_k) or self.tournament_k < 1:
            raise ValueError(
                f"tournament_k must be an int >= 1, got {self.tournament_k!r}")
        if not isinstance(self.pc, (int, float)) or not (0.0 <= self.pc <= 1.0):
            raise ValueError(f"pc must be a probability in [0, 1], got {self.pc!r}")
        if not isinstance(self.pm, (int, float)) or not (0.0 <= self.pm <= 1.0):
            raise ValueError(f"pm must be a probability in [0, 1], got {self.pm!r}")
        if self.nn_k != 8 or not _is_int(self.nn_k):
            raise ValueError(
                f"nn_k is frozen at 8 (the descent arithmetic "
                f"2 x (9 x 2 - 1) = 34 neighbours), got {self.nn_k!r}")
        if not _is_int(self.n_elite) or not (1 <= self.n_elite <= self.pop_size):
            raise ValueError(
                f"n_elite must be an int in [1, pop_size], got {self.n_elite!r}")
        if not isinstance(self.acceptance_margin, (int, float)) or \
                self.acceptance_margin < 0.0:
            raise ValueError(
                f"acceptance_margin must be a float >= 0 (0 = strictly "
                f"better; the margin only widens), got "
                f"{self.acceptance_margin!r}")
        if not isinstance(self.use_descent, bool):
            raise ValueError(
                f"use_descent must be a bool, got {self.use_descent!r}")
        if not _is_int(self.fence_rejection_cap) or self.fence_rejection_cap < 1:
            raise ValueError(
                f"fence_rejection_cap must be an int >= 1, got "
                f"{self.fence_rejection_cap!r}")

    def as_dict(self):
        """-> the full params dict (the run record's `params` field)."""
        return dataclasses.asdict(self)


# The registered parameter sets. These values were plumbing when registered;
# the pre-declared scoping sweep then selected exactly them as the stage-2
# frozen set, so they are the settings of every reported battery run. A
# default change is one committed edit with its rationale in the coding log.
PARAM_SETS = {
    "smoke": GAParams(
        pop_size=50,
        tournament_k=3,
        pc=0.9,
        pm=0.2,
        nn_k=8,
        n_elite=2,
        acceptance_margin=0.0,
        use_descent=True,
        fence_rejection_cap=1000,
    ),
}


def get_param_set(name):
    """-> the frozen GAParams of a registered set; fail hard on an
    unknown name (a typo must never run as a default)."""
    try:
        return PARAM_SETS[name]
    except KeyError:
        raise ValueError(
            f"unknown param set {name!r}; registered: {sorted(PARAM_SETS)}"
        ) from None
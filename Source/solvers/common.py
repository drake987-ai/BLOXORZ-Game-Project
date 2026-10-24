from dataclasses import dataclass

from engine.transitions import SearchAction
from models.level import Level
from models.state import GameState

# Risk-aware cost model shared by UCS and A* (see ucs_costs.md for the justification).
SWAP_COST = 0            # switching the active cube is free, as in the GUI's Moves counter
ROLL_COST = 1            # every roll or cube step
FRAGILE_PENALTY = 4      # any part of the block ends on a fragile tile


@dataclass(slots=True)
class SearchResult:
    path: list[SearchAction] | None
    search_time: float
    peak_memory: int
    expanded_nodes: int

    @property
    def solution_length(self) -> int | None:
        if self.path is None:
            return None
        return sum(action != "SWAP" for action in self.path)


def move_cost(level: Level, before: GameState, after: GameState, action: SearchAction) -> int:
    """Cost of one transition. Every non-SWAP move costs at least 1 and shifts the block's
    centre by at most 3 half-units, which is what keeps the A* heuristic admissible."""
    if action == "SWAP":
        return SWAP_COST
    cost = ROLL_COST
    if any(level.tile_at(r, c) == "F" for r, c in after.occupied_cells()):
        cost += FRAGILE_PENALTY
    return cost

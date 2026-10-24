"""Deterministic Bloxorz rules for ordinary and split-block gameplay."""

from typing import Literal

from models.level import Level, SplitSwitchSpec, SwitchSpec
from models.state import BlockState, GameState, Position

Action = Literal["UP", "DOWN", "LEFT", "RIGHT"]
SearchAction = Literal["UP", "DOWN", "LEFT", "RIGHT", "SWAP"]
ACTIONS: tuple[Action, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
DIRECTION: dict[Action, Position] = {
    "UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1),
}


def roll_block(block: BlockState, action: Action) -> BlockState:
    """Apply the original 12-case orientation transition table."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    r, c, o = block.row, block.col, block.orientation
    if o == "S":
        if action == "UP":
            return BlockState("V", r - 2, c)
        if action == "DOWN":
            return BlockState("V", r + 1, c)
        if action == "LEFT":
            return BlockState("H", r, c - 2)
        return BlockState("H", r, c + 1)
    if o == "H":
        if action == "UP":
            return BlockState("H", r - 1, c)
        if action == "DOWN":
            return BlockState("H", r + 1, c)
        if action == "LEFT":
            return BlockState("S", r, c - 1)
        return BlockState("S", r, c + 2)
    if action == "UP":
        return BlockState("S", r - 1, c)
    if action == "DOWN":
        return BlockState("S", r + 2, c)
    if action == "LEFT":
        return BlockState("V", r, c - 1)
    return BlockState("V", r, c + 1)


def is_legal(level: Level, block: BlockState, bridge_mask: int | None = None) -> bool:
    """A normal 1×1×2 block cannot stand on fragile floor."""
    if bridge_mask is None:
        bridge_mask = level.initial_bridge_mask
    if any(not level.is_supported(r, c, bridge_mask) for r, c in block.occupied_cells()):
        return False
    return not (block.orientation == "S" and level.tile_at(block.row, block.col) == "F")


def initial_state(level: Level) -> GameState:
    state = GameState(BlockState("S", *level.start), level.initial_bridge_mask)
    if not is_legal(level, state.block, state.bridge_mask):
        raise ValueError("Starting block has no valid supporting tile")
    return state


def _pressed(switch: SwitchSpec, cells: tuple[Position, ...], standing: bool) -> bool:
    if switch.kind == "soft":
        return switch.position in cells
    return standing and switch.position == cells[0]


def _apply_effect(level: Level, effect: str | None, targets: tuple[str, ...],
                  set_open: bool | None, mask: int) -> int:
    if effect is None:
        return mask
    for group_id in targets:
        bit = 1 << level.bridge_index(group_id)
        if effect == "toggle":
            mask ^= bit
        elif set_open:
            mask |= bit
        else:
            mask &= ~bit
    return mask


def _apply_switches(level: Level, before: tuple[Position, ...],
                    after: tuple[Position, ...], before_standing: bool,
                    after_standing: bool, mask: int) -> int:
    """Fire only on fresh presses; soft switches can also be pressed by cubes."""
    for switch in level.switches:
        if _pressed(switch, after, after_standing) and not _pressed(
            switch, before, before_standing
        ):
            mask = _apply_effect(level, switch.effect, switch.bridges, switch.set_open, mask)
    return mask


def _as_block_if_adjacent(a: Position, b: Position) -> BlockState | None:
    if a[0] == b[0] and abs(a[1] - b[1]) == 1:
        return BlockState("H", a[0], min(a[1], b[1]))
    if a[1] == b[1] and abs(a[0] - b[0]) == 1:
        return BlockState("V", min(a[0], b[0]), a[1])
    return None


def transition(level: Level, state: GameState, action: SearchAction) -> GameState | None:
    'Move, toggle newly pressed switches, check support; split/merge when appropriate.'
    if action == "SWAP":
        if state.cubes is None:
            return None
        return GameState(state.block, state.bridge_mask, state.cubes, 1 - state.active_cube)
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    if state.cubes is not None:
        before_cells = state.cubes
        moving = state.cubes[state.active_cube]
        dr, dc = DIRECTION[action]
        new_pos = (moving[0] + dr, moving[1] + dc)
        if new_pos == state.cubes[1 - state.active_cube]:
            return None
        if not level.is_supported(*new_pos, state.bridge_mask):
            return None
        updated = list(state.cubes)
        updated[state.active_cube] = new_pos
        after_cells = tuple(updated)
        merged = _as_block_if_adjacent(after_cells[0], after_cells[1])
        mask = _apply_switches(
            level, before_cells, after_cells, False, False, state.bridge_mask
        )
        if not all(level.is_supported(*pos, mask) for pos in after_cells):
            return None
        if merged is not None:
            if not is_legal(level, merged, mask):
                return None
            return GameState(merged, mask)
        return GameState(state.block, mask, (after_cells[0], after_cells[1]), state.active_cube)

    before_block = state.block
    candidate = roll_block(before_block, action)
    if not is_legal(level, candidate, state.bridge_mask):
        return None
    mask = _apply_switches(
        level, before_block.occupied_cells(), candidate.occupied_cells(),
        before_block.orientation == "S", candidate.orientation == "S", state.bridge_mask
    )

    split: SplitSwitchSpec | None = None
    if candidate.orientation == "S" and candidate.occupied_cells() != before_block.occupied_cells():
        split = level.split_at(candidate.row, candidate.col)
    if split is not None:
        mask = _apply_effect(level, split.effect, split.bridges, split.set_open, mask)

        mask = _apply_switches(
            level, candidate.occupied_cells(), split.cubes, True, False, mask
        )
        if not all(level.is_supported(*p, mask) for p in split.cubes):
            return None

        merged = _as_block_if_adjacent(*split.cubes)
        if merged is not None:
            if not is_legal(level, merged, mask):
                return None
            return GameState(merged, mask)
        return GameState(candidate, mask, split.cubes, 0)
    if not is_legal(level, candidate, mask):
        return None
    return GameState(candidate, mask)


def successors(level: Level, state: GameState) -> list[tuple[SearchAction, GameState]]:
    choices: tuple[SearchAction, ...] = ACTIONS + (("SWAP",) if state.is_split else ())
    moves: list[tuple[SearchAction, GameState]] = []
    for action in choices:
        result = transition(level, state, action)
        if result is not None:
            moves.append((action, result))
    return moves


def is_goal(level: Level, state: GameState) -> bool:
    return (state.cubes is None and state.block.orientation == "S"
            and (state.block.row, state.block.col) == level.goal)


def step_cost(before: GameState, after: GameState, action: SearchAction) -> int:
    'Nonuniform UCS/A* cost: bridge-changing action costs 2; else 1.'
    return 2 if before.bridge_mask != after.bridge_mask else 1

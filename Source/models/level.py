"""Validated Bloxorz maps, bridges, soft/heavy switches and split switches."""

from dataclasses import dataclass
import json
from pathlib import Path

from models.state import Position

ALLOWED_TILES = frozenset({"#", ".", "S", "G", "F", "B", "o", "X", "P"})


@dataclass(frozen=True, slots=True)
class BridgeGroup:
    """Bridge cells sharing one open/closed state bit."""

    id: str
    cells: tuple[Position, ...]
    initially_open: bool = False


@dataclass(frozen=True, slots=True)
class SwitchSpec:
    """A switch acting on one or more named bridge groups."""

    position: Position
    kind: str
    effect: str
    bridges: tuple[str, ...]
    set_open: bool | None = None


@dataclass(frozen=True, slots=True)
class SplitSwitchSpec:
    """Standing on P teleports the block into two separate cubes."""
    position: Position
    cubes: tuple[Position, Position]
    effect: str | None = None
    bridges: tuple[str, ...] = ()
    set_open: bool | None = None


@dataclass(frozen=True, slots=True)
class Level:
    name: str
    rows: tuple[str, ...]
    start: Position
    goal: Position
    bridges: tuple[BridgeGroup, ...] = ()
    switches: tuple[SwitchSpec, ...] = ()
    split_switches: tuple[SplitSwitchSpec, ...] = ()

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def initial_bridge_mask(self) -> int:
        mask = 0
        for index, group in enumerate(self.bridges):
            if group.initially_open:
                mask |= 1 << index
        return mask

    def tile_at(self, row: int, col: int) -> str:
        if not (0 <= row < self.height and 0 <= col < self.width):
            return "#"
        return self.rows[row][col]

    def bridge_index_at(self, row: int, col: int) -> int | None:
        for index, group in enumerate(self.bridges):
            if (row, col) in group.cells:
                return index
        return None

    def bridge_index(self, bridge_id: str) -> int:
        for index, group in enumerate(self.bridges):
            if group.id == bridge_id:
                return index
        raise ValueError(f"Unknown bridge id: {bridge_id!r}")

    def split_at(self, row: int, col: int) -> SplitSwitchSpec | None:
        for spec in self.split_switches:
            if spec.position == (row, col):
                return spec
        return None

    def is_supported(self, row: int, col: int, bridge_mask: int = 0) -> bool:
        tile = self.tile_at(row, col)
        if tile == "#":
            return False
        if tile == "B":
            index = self.bridge_index_at(row, col)
            return index is not None and bool(bridge_mask & (1 << index))
        return True


def _position(value: object, label: str, rows: tuple[str, ...]) -> Position:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"{label} must be [row, col] with integer coordinates")
    row, col = value
    if not (0 <= row < len(rows) and 0 <= col < len(rows[0])):
        raise ValueError(f"{label} is outside the level: {(row, col)}")
    return row, col


def load_level(path: str | Path) -> Level:
    """Load and validate a level, including bridges and switches."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Level JSON must be an object")
    name = data.get("name", "Unnamed level")
    raw_rows = data.get("map")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' must be a nonempty string")
    if not isinstance(raw_rows, list) or not raw_rows or not all(
        isinstance(row, str) for row in raw_rows
    ):
        raise ValueError("'map' must be a nonempty list of strings")
    if not raw_rows[0] or any(len(row) != len(raw_rows[0]) for row in raw_rows):
        raise ValueError("Every row must have the same nonzero length")

    rows = tuple(raw_rows)
    starts: list[Position] = []
    goals: list[Position] = []
    bridge_cells: set[Position] = set()
    switch_cells: set[Position] = set()
    split_cells: set[Position] = set()
    for row, line in enumerate(rows):
        for col, tile in enumerate(line):
            if tile not in ALLOWED_TILES:
                raise ValueError(f"Unsupported tile {tile!r} at {(row, col)}")
            if tile == "S":
                starts.append((row, col))
            elif tile == "G":
                goals.append((row, col))
            elif tile == "B":
                bridge_cells.add((row, col))
            elif tile in ("o", "X"):
                switch_cells.add((row, col))
            elif tile == "P":
                split_cells.add((row, col))
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("Every level must have exactly one S and one G")

    bridge_defs = data.get("bridges", [])
    switch_defs = data.get("switches", [])
    split_defs = data.get("split_switches", [])
    if not all(isinstance(x, list) for x in (bridge_defs, switch_defs, split_defs)):
        raise ValueError("'bridges', 'switches', 'split_switches' must be lists")

    groups: list[BridgeGroup] = []
    used_bridge_cells: set[Position] = set()
    used_ids: set[str] = set()
    for index, entry in enumerate(bridge_defs):
        if not isinstance(entry, dict):
            raise ValueError(f"bridges[{index}] must be an object")
        identifier = entry.get("id")
        positions = entry.get("cells")
        initially_open = entry.get("initially_open", False)
        if not isinstance(identifier, str) or not identifier.strip() or identifier in used_ids:
            raise ValueError(f"Invalid or repeated bridge id: {identifier!r}")
        if type(initially_open) is not bool:
            raise ValueError("'initially_open' must be a boolean")
        if not isinstance(positions, list) or not positions:
            raise ValueError(f"Bridge {identifier!r} must have a nonempty cells list")
        cells = tuple(_position(p, f"bridge {identifier!r} cell", rows) for p in positions)
        if len(set(cells)) != len(cells) or any(cell in used_bridge_cells for cell in cells):
            raise ValueError("Bridge cells cannot be duplicated or shared between groups")
        if any(rows[r][c] != "B" for r, c in cells):
            raise ValueError(f"Bridge {identifier!r} references a tile not marked 'B'")
        used_ids.add(identifier)
        used_bridge_cells.update(cells)
        groups.append(BridgeGroup(identifier, cells, initially_open))
    if used_bridge_cells != bridge_cells:
        raise ValueError("Every 'B' tile must belong to exactly one bridge group")

    switches: list[SwitchSpec] = []
    used_switch_cells: set[Position] = set()
    for index, entry in enumerate(switch_defs):
        if not isinstance(entry, dict):
            raise ValueError(f"switches[{index}] must be an object")
        position = _position(entry.get("position"), f"switches[{index}].position", rows)
        kind = entry.get("type")
        effect = entry.get("effect")
        targets = entry.get("bridges")
        if kind not in ("soft", "heavy") or effect not in ("toggle", "permanent"):
            raise ValueError("Switch type must be soft/heavy and effect toggle/permanent")
        if rows[position[0]][position[1]] != ("o" if kind == "soft" else "X"):
            raise ValueError(f"Switch {position} has wrong map symbol")
        if position in used_switch_cells:
            raise ValueError(f"Multiple switches at {position}")
        if not isinstance(targets, list) or not targets or not all(
            isinstance(target, str) and target in used_ids for target in targets
        ) or len(set(targets)) != len(targets):
            raise ValueError(f"Switch {position} must target distinct existing bridge ids")
        set_open = entry.get("set_open")
        if effect == "permanent":
            if type(set_open) is not bool:
                raise ValueError(f"Permanent switch {position} needs boolean set_open")
        elif set_open is not None:
            raise ValueError(f"Toggle switch {position} cannot set set_open")
        used_switch_cells.add(position)
        switches.append(SwitchSpec(position, kind, effect, tuple(targets), set_open))
    if used_switch_cells != switch_cells:
        raise ValueError("Every 'o' or 'X' tile must have one matching switch definition")

    splits: list[SplitSwitchSpec] = []
    used_split_cells: set[Position] = set()
    for index, entry in enumerate(split_defs):
        if not isinstance(entry, dict):
            raise ValueError(f"split_switches[{index}] must be an object")
        position = _position(entry.get("position"), f"split_switches[{index}].position", rows)
        if rows[position[0]][position[1]] != "P" or position in used_split_cells:
            raise ValueError(f"Invalid or duplicate split switch at {position}")
        destinations = entry.get("cubes")
        if not isinstance(destinations, list) or len(destinations) != 2:
            raise ValueError(f"Split switch at {position} must define two destination cubes")
        p0, p1 = (
            _position(p, f"split switch destination {j}", rows)
            for j, p in enumerate(destinations)
        )
        if p0 == p1 or any(rows[r][c] == "#" for r, c in (p0, p1)):
            raise ValueError(f"Split destinations must be distinct non-void cells: {position}")
        effect = entry.get("effect")
        targets = entry.get("bridges", [])
        set_open = entry.get("set_open")
        if effect not in (None, "toggle", "permanent"):
            raise ValueError("Split effect must be toggle/permanent or omitted")
        if not isinstance(targets, list) or not all(
            isinstance(target, str) and target in used_ids for target in targets
        ) or len(set(targets)) != len(targets):
            raise ValueError("Split switch refers to unknown or duplicate bridge IDs")
        if (effect is None and (targets or set_open is not None)) or (
            effect is not None and not targets
        ):
            raise ValueError("Split effect and bridge targets must be configured together")
        if effect == "permanent" and type(set_open) is not bool:
            raise ValueError("Permanent split switch must define boolean set_open")
        if effect != "permanent" and set_open is not None:
            raise ValueError("Only permanent split switches may set set_open")
        used_split_cells.add(position)
        splits.append(SplitSwitchSpec(position, (p0, p1), effect, tuple(targets), set_open))
    if used_split_cells != split_cells:
        raise ValueError("Every 'P' tile must define exactly one split switch")
    return Level(name, rows, starts[0], goals[0], tuple(groups), tuple(switches), tuple(splits))

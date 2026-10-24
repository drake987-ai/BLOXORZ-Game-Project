"""Hashable, immutable Bloxorz states for normal and split-block modes."""

from dataclasses import dataclass
from typing import Literal

Orientation = Literal["S", "H", "V"]
Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BlockState:
    """S: one tile; H: left tile of a horizontal block; V: top tile."""
    orientation: Orientation
    row: int
    col: int

    def __post_init__(self) -> None:
        if self.orientation not in ("S", "H", "V"):
            raise ValueError(f"Invalid orientation: {self.orientation!r}")

    def occupied_cells(self) -> tuple[Position, ...]:
        if self.orientation == "S":
            return ((self.row, self.col),)
        if self.orientation == "H":
            return ((self.row, self.col), (self.row, self.col + 1))
        return ((self.row, self.col), (self.row + 1, self.col))


@dataclass(frozen=True, slots=True)
class GameState:
    'Full search state: block, bridge bits, and optional independently moving cubes.'
    block: BlockState
    bridge_mask: int = 0
    cubes: tuple[Position, Position] | None = None
    active_cube: int = 0

    def __post_init__(self) -> None:
        if self.bridge_mask < 0:
            raise ValueError("Negative bridge mask")
        if self.active_cube not in (0, 1):
            raise ValueError("Active cube must be 0 or 1")
        if self.cubes is None and self.active_cube != 0:
            raise ValueError("Unsplit block must have active_cube=0")
        if self.cubes is not None and self.cubes[0] == self.cubes[1]:
            raise ValueError("Split cubes cannot overlap")

    @property
    def is_split(self) -> bool:
        return self.cubes is not None

    def occupied_cells(self) -> tuple[Position, ...]:
        return self.cubes if self.cubes is not None else self.block.occupied_cells()

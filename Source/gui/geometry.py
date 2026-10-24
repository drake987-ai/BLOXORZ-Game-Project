"""Pure geometry conversion: grid cells/block states to 3D world coordinates.

This module intentionally has no Panda3D dependency, making it testable in
any environment, including systems without a graphical display.
"""

from dataclasses import dataclass

from models.state import BlockState


@dataclass(frozen=True, slots=True)
class BoxPose:
    """Center (x, y, z) and dimensions (sx, sy, sz) of a 3D box."""

    x: float
    y: float
    z: float
    sx: float
    sy: float
    sz: float


def tile_to_world(row: int, col: int) -> tuple[float, float, float]:
    """Map column to +X, row to -Y, and keep the board surface at Z=0."""
    return (float(col), float(-row), 0.0)


def block_to_world(block: BlockState) -> BoxPose:
    """Place the 1x1x2 block on top of tiles, centered on its occupied cells.

    The 0.92 scale avoids adjacent meshes visually merging; it does not change
    game rules, since the engine still uses the exact logical grid cells.
    """
    r, c = block.row, block.col
    if block.orientation == "S":
        return BoxPose(float(c), float(-r), 0.92, 0.92, 0.92, 1.84)
    if block.orientation == "H":
        return BoxPose(c + 0.5, float(-r), 0.46, 1.84, 0.92, 0.92)
    return BoxPose(float(c), -r - 0.5, 0.46, 0.92, 1.84, 0.92)

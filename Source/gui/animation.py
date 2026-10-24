"Panda3D animations for block and split-cube movement."

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from gui.geometry import BoxPose, block_to_world
from models.state import GameState

ROLL_SECONDS = 0.24
SLIDE_SECONDS = 0.19
SPLIT_SECONDS = 0.35
MERGE_SECONDS = 0.18
TASK_NAME = "bloxorz-player-animation"


def smoothstep(t: float) -> float:
    """Cubic ease-in/ease-out, with no abrupt start or stop."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def roll_pivot(before: BoxPose, after: BoxPose, action: str) -> tuple[float, float, float]:
    'A pivot making a quarter-turn join the *actual* scaled visual centers.'
    if action in ("RIGHT", "LEFT"):
        a, b = before.x, after.x
        s = 1 if action == "RIGHT" else -1
        d1 = b - s * before.z
        d2 = after.z + s * a
        ph = (d1 + s * d2) / 2.0
        pz = (d2 - s * d1) / 2.0
        return ph, before.y, pz
    if action in ("UP", "DOWN"):
        a, b = before.y, after.y
        s = 1 if action == "UP" else -1
        d1 = b - s * before.z
        d2 = after.z + s * a
        ph = (d1 + s * d2) / 2.0
        pz = (d2 - s * d1) / 2.0
        return before.x, ph, pz
    raise ValueError(f"Not a rolling action: {action!r}")


class Animator:
    'Runs one valid move at a time using a Panda3D Task per animation.'

    def __init__(self, app: Any, vec3: Any, quat: Any, clock: Any) -> None:
        self.app = app
        self._vec3 = vec3
        self._quat = quat
        self._clock = clock
        self.active = False
        self._phase_time = 0.0
        self._phases: list[tuple[float, Callable[[float], None], Callable[[], None]]] = []
        self._pivot: Any = None
        self._phase_index = 0
        self._on_done: Callable[[], None] | None = None

    def cancel(self) -> None:
        """Discard an in-flight visual transition before restart/level change."""
        if self.active:
            self.app.taskMgr.remove(TASK_NAME)
        self.active = False
        self._phases.clear()
        self._phase_index = 0
        self._phase_time = 0.0
        self._on_done = None
        self._discard_pivot()

    def _discard_pivot(self) -> None:
        if self._pivot is not None:

            self._pivot.removeNode()
            self._pivot = None
            self.app._block_visual = None

    def _add_phase(
        self, duration: float, update: Callable[[float], None], finish: Callable[[], None]
    ) -> None:
        self._phases.append((duration, update, finish))

    def start(
        self, before: GameState, after: GameState, action: str,
        on_done: Callable[[], None] | None = None,
    ) -> bool:
        """Prepare visuals, return False if the caller should refresh instantly.

        `on_done` replaces the final visual refresh (used for the fall animation).
        """
        if action == "SWAP" or self.active:
            return False
        if before.cubes is None and self.app._block_visual is None:
            return False
        if before.cubes is not None and len(self.app._cube_visuals) != 2:
            return False

        self._phases = []
        self._phase_index = 0
        self._phase_time = 0.0
        if before.cubes is None:
            self._prepare_roll(before, after, action)
            if after.cubes is not None:
                self._prepare_split(after)
        else:
            self._prepare_cube_slide(before, after)
            if after.cubes is None:
                self._prepare_merge(before, after)
        if not self._phases:
            return False
        self.active = True
        self._on_done = on_done
        self._phases[0][1](0.0)
        self.app.taskMgr.add(self._tick, TASK_NAME)
        return True

    def _prepare_roll(self, before: GameState, after: GameState, action: str) -> None:
        before_pose = block_to_world(before.block)
        after_pose = block_to_world(after.block)
        pivot_position = roll_pivot(before_pose, after_pose, action)
        pivot = self.app.render.attachNewNode("animation-roll-pivot")
        pivot.setPos(*pivot_position)
        self.app._block_visual.wrtReparentTo(pivot)
        self._pivot = pivot
        if action in ("RIGHT", "LEFT"):
            axis = self._vec3(0, 1, 0)
            degrees = 90 if action == "RIGHT" else -90
        else:
            axis = self._vec3(1, 0, 0)
            degrees = -90 if action == "UP" else 90

        def update(t: float) -> None:
            rotation = self._quat()
            rotation.setFromAxisAngle(degrees * smoothstep(t), axis)
            pivot.setQuat(rotation)

        def finish() -> None:

            if after.cubes is not None:
                self._discard_pivot()

        self._add_phase(ROLL_SECONDS, update, finish)

    def _prepare_split(self, after: GameState) -> None:
        assert after.cubes is not None

        origin = block_to_world(after.block)
        starts = ((origin.x, origin.y, 0.46),) * 2
        targets = tuple((float(c), float(-r), 0.46) for r, c in after.cubes)
        spawned: list[Any] = []

        def update(t: float) -> None:
            if not spawned:
                for i in range(2):
                    color = ((1, 0.85, 0.28, 1) if i == after.active_cube
                             else (0.77, 0.49, 0.20, 1))
                    node = self.app._place_box(
                        BoxPose(*starts[i], 0.92, 0.92, 0.92),
                        color, f"splitting-cube-{i+1}",
                    )
                    spawned.append(node)
                self.app._cube_visuals = spawned
            eased = smoothstep(t)
            bounce = 0.72 * math.sin(math.pi * t)
            for i, node in enumerate(spawned):
                start = starts[i]
                dest = targets[i]
                node.setPos(
                    start[0] + (dest[0] - start[0]) * eased,
                    start[1] + (dest[1] - start[1]) * eased,
                    0.46 + bounce,
                )

        self._add_phase(SPLIT_SECONDS, update, lambda: None)

    def _prepare_cube_slide(self, before: GameState, after: GameState) -> None:
        assert before.cubes is not None
        i = before.active_cube
        row, col = before.cubes[i]
        start = (float(col), float(-row), 0.46)
        if after.cubes is not None:
            nr, nc = after.cubes[i]
        else:


            target_cells = set(after.block.occupied_cells())
            target_cells.discard(before.cubes[1 - i])
            if len(target_cells) != 1:
                raise ValueError("Merged block must contain the stationary cube")
            nr, nc = target_cells.pop()
        dest = (float(nc), float(-nr), 0.46)
        node = self.app._cube_visuals[i]

        def update(t: float) -> None:
            eased = smoothstep(t)
            node.setPos(
                start[0] + (dest[0] - start[0]) * eased,
                start[1] + (dest[1] - start[1]) * eased,
                0.46 + 0.09 * math.sin(math.pi * t),
            )

        self._add_phase(SLIDE_SECONDS, update, lambda: None)

    def _prepare_merge(self, before: GameState, after: GameState) -> None:
        pose = block_to_world(after.block)
        assert before.cubes is not None
        nodes = self.app._cube_visuals


        def update(t: float) -> None:
            pulse = math.sin(math.pi * t)
            for i, node in enumerate(nodes):
                row_col = merged_positions[i]
                x, y = float(row_col[1]), float(-row_col[0])
                node.setPos(
                    x + (pose.x - x) * 0.09 * pulse,
                    y + (pose.y - y) * 0.09 * pulse,
                    0.46 + 0.13 * pulse,
                )


        active = before.active_cube
        stationary = before.cubes[1 - active]
        moving = (set(after.block.occupied_cells()) - {stationary}).pop()
        merged_positions = list(before.cubes)
        merged_positions[active] = moving
        self._add_phase(MERGE_SECONDS, update, lambda: None)

    def _tick(self, task: Any) -> Any:
        dt = min(max(float(self._clock.getGlobalClock().getDt()), 0.0), 0.1)
        self._phase_time += dt
        while self._phase_index < len(self._phases):
            duration, update, finish = self._phases[self._phase_index]
            progress = min(1.0, self._phase_time / duration)
            update(progress)
            if progress < 1.0:
                return task.cont
            finish()
            self._phase_time -= duration
            self._phase_index += 1
            if self._phase_index < len(self._phases):
                self._phases[self._phase_index][1](0.0)
        self.active = False
        self._discard_pivot()
        self._phases.clear()
        done, self._on_done = self._on_done, None
        if done is not None:
            done()
        else:
            self.app._refresh_block()
        return task.done

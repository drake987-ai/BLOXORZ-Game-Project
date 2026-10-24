"""Bloxorz 3D GUI; mechanics stay in engine/transitions.py."""

import math
from pathlib import Path

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame
from direct.gui.OnscreenText import OnscreenText
from direct.interval.IntervalGlobal import (
    Func, LerpColorScaleInterval, LerpHprInterval, LerpPosInterval, LerpScaleInterval,
    Parallel, Sequence, Wait,
)
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    DirectionalLight,
    Filename,
    TextNode,
    Vec4,
    WindowProperties,
    ClockObject,
    Quat,
    Vec3,
    Plane,
    PlaneNode,
    Point3,
)

from engine.transitions import (
    Action, SearchAction, initial_state, is_goal, is_legal, roll_block, transition,
)
from gui.geometry import BoxPose, block_to_world, tile_to_world
from gui.meshes import bridge_model, switch_marker, unit_box
from gui.animation import Animator
from models.catalog import LevelEntry
from models.level import Level, load_level
from models.state import GameState

from solvers.bfs import bfs
from solvers.dfs import dfs
from solvers.ucs import ucs
from solvers.astar import astar
from solvers.common import SearchResult

FONT_DIR = Path(__file__).resolve().parent / "fonts"

# Shared UI palette (RGBA).
BUTTON_COLOR = (0.12, 0.35, 0.55, 0.96)
ACTIVE_LEVEL_COLOR = (0.13, 0.52, 0.40, 1)
ACTIVE_LEVEL_BORDER = (0.98, 0.82, 0.30, 1)
PANEL_COLOR = (0.055, 0.085, 0.14, 0.92)
TEXT_COLOR = (0.85, 0.9, 0.98, 1)
OVERLAY_COLOR = (0.02, 0.03, 0.05, 0.85)
CAMERA_SPIN_SPEED = 90.0  # degrees per second while Q/E is held
FALL_DROP = 10.0          # world units the block falls after an invalid move
FALL_SECONDS = 1.0
FALL_TUMBLE = 70.0        # degrees of tumble while falling
FALL_HEADING = 20.0
SINK_DROP = 2.0           # just over the block's 1.84 height, so it ends fully below the floor
SINK_SECONDS = 1.0


class BloxorzApp(ShowBase):
    """3D renderer and controller."""

    def __init__(
        self, level: Level, level_entries: tuple[LevelEntry, ...] = ()
    ) -> None:
        super().__init__()
        self._font = self._load_font("Poppins-Regular.ttf")
        self._font_bold = self._load_font("Poppins-SemiBold.ttf")
        self.level = level
        self.state: GameState = initial_state(level)
        self.steps = 0
        self._solution_running = False
        self.game_over = False
        self._fall_landing: GameState | None = None
        self._fall_sequence = None
        self._fall_action: str = "UP"
        self._win_sequence = None
        self._sink_clip = None
        self._win_popup_sequence = None
        self._playback_path: list[SearchAction] = []
        self._playback_states: list[GameState] = []
        self._playback_moves: list[int] = []
        self._playback_index = 0
        self._playback_base_steps = 0
        self._playback_name = ""
        self._unit_box = unit_box()
        self._block_visual = None
        self._cube_visuals = []
        self._bridge_visuals = {}
        self._tile_visuals = []
        self._level_entries = level_entries
        self._selector_open = False
        self._selector_page = 0
        self._selector_widgets = []
        self._selector_panel = None
        self._selector_overlay = None
        self._current_level_index = next(
            (i for i, e in enumerate(level_entries) if e.name == level.name), -1
        )

        self._animator = Animator(self, Vec3, Quat, ClockObject)

        props = WindowProperties()
        props.setTitle("Bloxorz")
        props.setSize(1200, 760)
        props.setForeground(True)
        if self.win is None:
            raise RuntimeError(
                "Panda3D could not open a graphics window. "
                "Check your display and graphics drivers."
            )
        self.win.requestProperties(props)

        self.disableMouse()
        self.setBackgroundColor(0.075, 0.105, 0.16)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self._setup_lights()
        self._setup_camera()
        self.taskMgr.add(self._camera_drag_task, "camera-drag")
        self.taskMgr.add(self._camera_spin_task, "camera-spin")
        self._draw_level()
        self._setup_ui()
        self._bind_keys()
        self._refresh_block()
        if self.level.width >= 50 or (
            self._level_entries and
            (self.level.width >= 35 or self.level.height >= 25)
        ):
            self._focus_player()


    def _load_font(self, filename: str):
        """Load a bundled Poppins face with a crisper glyph atlas."""
        font = self.loader.loadFont(
            Filename.fromOsSpecific(str(FONT_DIR / filename)).getFullpath()
        )
        font.setPixelsPerUnit(64)
        return font

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.58, 0.62, 0.72, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.86, 0.85, 0.79, 1))
        sunlight = self.render.attachNewNode(sun)
        sunlight.setHpr(-35, -55, 0)
        self.render.setLight(sunlight)

    def _setup_camera(self) -> None:
        self._camera_default_center = (
            (self.level.width - 1) / 2,
            -(self.level.height - 1) / 2,
        )
        self._camera_default_distance = max(
            9.0, self.level.width * 1.5, self.level.height * 2.0
        )
        self._camera_center_x, self._camera_center_y = self._camera_default_center
        self._camera_distance = self._camera_default_distance
        self._camera_drag_last = None
        self._camera_yaw = 0.0
        self._camera_spin = 0
        self.camLens.setFov(45)
        self.camLens.setNearFar(0.1, 1000)
        self._apply_camera()

    def _apply_camera(self) -> None:
        """Keep a fixed view angle while changing camera target or zoom."""
        x, y, d = self._camera_center_x, self._camera_center_y, self._camera_distance
        yaw = math.radians(self._camera_yaw)
        dx, dy = d * 0.75, -d
        self.camera.setPos(
            x + dx * math.cos(yaw) - dy * math.sin(yaw),
            y + dx * math.sin(yaw) + dy * math.cos(yaw),
            d * 0.95,
        )
        self.camera.lookAt(x, y, 0.0)

    def _zoom_camera(self, factor: float) -> None:
        """Mouse wheel changes the distance, not the game board or its rules."""
        self._camera_distance = max(
            5.0,
            min(self._camera_default_distance * 3.0, self._camera_distance * factor),
        )
        self._apply_camera()

    def _set_camera_spin(self, direction: int) -> None:
        """Q/E hold-to-rotate (keyboard alternative for trackpads)."""
        self._camera_spin = direction

    def _camera_spin_task(self, task):
        if self._camera_spin:
            dt = ClockObject.getGlobalClock().getDt()
            self._camera_yaw += self._camera_spin * CAMERA_SPIN_SPEED * dt
            self._apply_camera()
        return task.cont

    def _reset_camera(self) -> None:
        """Return to a whole-level view (shortcut C)."""
        self._camera_yaw = 0.0
        self._camera_center_x, self._camera_center_y = self._camera_default_center
        self._camera_distance = self._camera_default_distance
        self._camera_drag_last = None
        self._apply_camera()

    def _focus_player(self) -> None:
        """Center the view on the player's current logical position (shortcut F)."""
        if self.state.cubes is None:
            pose = block_to_world(self.state.block)
            self._camera_center_x, self._camera_center_y = pose.x, pose.y
        else:
            row, col = self.state.cubes[self.state.active_cube]
            self._camera_center_x, self._camera_center_y = float(col), float(-row)
        self._camera_distance = min(self._camera_distance, 18.0)
        self._camera_drag_last = None
        self._apply_camera()

    def _start_camera_drag(self) -> None:
        """Use RIGHT mouse drag to pan without affecting the gameplay keys."""
        if self.mouseWatcherNode.hasMouse():
            mouse = self.mouseWatcherNode.getMouse()
            self._camera_drag_last = (mouse.getX(), mouse.getY())

    def _stop_camera_drag(self) -> None:
        self._camera_drag_last = None

    def _camera_drag_task(self, task):
        """Pan in camera screen directions while the right mouse button is held."""
        if self._camera_drag_last is None:
            return task.cont
        if not self.mouseWatcherNode.hasMouse():
            self._stop_camera_drag()
            return task.cont

        mouse = self.mouseWatcherNode.getMouse()
        next_mouse = (mouse.getX(), mouse.getY())
        dx = next_mouse[0] - self._camera_drag_last[0]
        dy = next_mouse[1] - self._camera_drag_last[1]
        self._camera_drag_last = next_mouse
        if dx or dy:
            orientation = self.camera.getQuat(self.render)
            right = orientation.getRight()
            up = orientation.getUp()
            speed = self._camera_distance * 0.65
            self._camera_center_x -= speed * (right.getX() * dx + up.getX() * dy)
            self._camera_center_y -= speed * (right.getY() * dx + up.getY() * dy)
            self._apply_camera()
        return task.cont

    def _place_box(
        self, pose: BoxPose, color: tuple[float, float, float, float], name: str
    ):
        box = self._unit_box.copyTo(self.render)
        box.setName(name)
        box.setPos(pose.x, pose.y, pose.z)
        box.setScale(pose.sx, pose.sy, pose.sz)
        box.setColor(*color)
        return box

    def _draw_level(self) -> None:
        palette = {
            ".": (0.45, 0.59, 0.72, 1),
            "S": (0.25, 0.72, 0.78, 1),
            "G": (0.19, 0.16, 0.31, 1),
            "F": (0.93, 0.50, 0.19, 1),
        }
        for row, line in enumerate(self.level.rows):
            for col, tile in enumerate(line):
                if tile in ("#", "B"):
                    continue
                x, y, _ = tile_to_world(row, col)
                thickness = 0.10 if tile == "G" else 0.18
                self._tile_visuals.append(self._place_box(
                    BoxPose(x, y, -thickness / 2, 0.97, 0.97, thickness),
                    palette.get(tile, palette["."]),
                    f"tile-{row}-{col}",
                ))
                if tile in ("o", "X", "P"):
                    marker = switch_marker(tile, self._unit_box)
                    marker.reparentTo(self.render)
                    marker.setPos(x, y, 0)
                    self._tile_visuals.append(marker)

    def _refresh_bridges(self) -> None:
        """Show open bridge tiles and remove closed ones."""
        for index, group in enumerate(self.level.bridges):
            opened = bool(self.state.bridge_mask & (1 << index))
            for row, col in group.cells:
                pos = (row, col)
                visual = self._bridge_visuals.get(pos)
                if opened and visual is None:
                    x, y, _ = tile_to_world(row, col)
                    panel = bridge_model(self._unit_box)
                    panel.reparentTo(self.render)
                    panel.setPos(x, y, 0)
                    self._bridge_visuals[pos] = panel
                elif not opened and visual is not None:
                    visual.removeNode()
                    del self._bridge_visuals[pos]

    def _text(
        self, text: str, pos: tuple[float, float], scale: float, parent=None,
        fg=TEXT_COLOR, align=TextNode.ACenter, bold: bool = False, shadow=None,
    ) -> OnscreenText:
        return OnscreenText(
            text=text, pos=pos, scale=scale, fg=fg, align=align, shadow=shadow,
            parent=parent if parent is not None else self.aspect2d,
            font=self._font_bold if bold else self._font, mayChange=True,
        )

    def _setup_ui(self) -> None:
        self._title = self._text(
            f"BLOXORZ | {self.level.name}", (0, 0.91), 0.072,
            fg=(0.96, 0.98, 1, 1), bold=True, shadow=(0, 0, 0, 0.75),
        )
        self._info = self._text("", (0, 0.79), 0.05)
        self._status = self._text("", (0, -0.78), 0.045, fg=(0.97, 0.9, 0.66, 1))
        self._setup_control_bar()
        self._setup_analytics_panel()
        self._setup_playback_controls()
        self._setup_help_overlay()
        self._setup_game_over_panel()
        self._setup_win_popup()

        if self._level_entries:
            self._setup_level_selector()

    def _setup_control_bar(self) -> None:
        """Game controls cluster bottom-left, solver controls bottom-right."""
        left, right = self.a2dBottomLeft, self.a2dBottomRight
        for parent, frame in ((left, (0.02, 0.84)), (right, (-1.10, -0.02))):
            DirectFrame(
                parent=parent, pos=(0, 0, 0.02), relief=DGG.FLAT,
                frameSize=(frame[0], frame[1], 0, 0.14), frameColor=(0.04, 0.06, 0.10, 0.55),
            )
        self._make_button("New Game", (0.25, 0, 0.065), self.new_game, left, 6.0)
        self._make_button("Restart", (0.62, 0, 0.065), self.restart_level, left, 5.0)
        solvers = (
            ("BFS", self.solve_bfs), ("DFS", self.solve_dfs),
            ("UCS", self.solve_ucs), ("A*", self.solve_astar),
        )
        for i, (label, command) in enumerate(solvers):
            x = -0.15 - 0.255 * (len(solvers) - 1 - i)
            self._make_button(label, (x, 0, 0.065), command, right, 3.6)

    @staticmethod
    def _state_colors(base: tuple[float, float, float, float]):
        """DirectButton frame colours: normal, pressed, hovered, disabled."""
        r, g, b, a = base
        return (
            base,
            (r * 0.75, g * 0.75, b * 0.75, a),
            (min(r * 1.25 + 0.04, 1), min(g * 1.25 + 0.04, 1), min(b * 1.25 + 0.04, 1), a),
            (0.22, 0.25, 0.30, 0.7),
        )

    def _make_button(
        self, label: str, pos: tuple[float, float, float], command, parent=None,
        width: float = 4.0, scale: float = 0.06, color=BUTTON_COLOR, extra_args=(),
        text_scale: float = 1.0,
    ) -> DirectButton:
        """Uniform button: `width` is in text units, so it scales with `scale`.

        Poppins caps sit about 0.35 above the baseline, so the frame is offset to
        centre the label vertically.
        """
        return DirectButton(
            parent=parent if parent is not None else self.aspect2d,
            text=label, text_font=self._font_bold, text_scale=text_scale,
            text_fg=(1, 1, 1, 1), pos=pos, scale=scale,
            frameSize=(-width / 2, width / 2, -0.4, 1.1),
            frameColor=self._state_colors(color), relief=DGG.FLAT,
            command=command, extraArgs=list(extra_args),
        )

    def _setup_analytics_panel(self) -> None:
        """Right-hand metrics card, shown after a solver finishes."""
        self._analytics = DirectFrame(
            parent=self.a2dTopRight, pos=(-0.04, 0, -0.30), relief=DGG.FLAT,
            frameSize=(-0.80, 0, -0.64, 0), frameColor=PANEL_COLOR,
        )
        self._analytics_title = self._text(
            "", (-0.76, -0.10), 0.05, self._analytics,
            fg=(0.97, 0.9, 0.66, 1), align=TextNode.ALeft, bold=True,
        )
        self._analytics_outcome = self._text(
            "", (-0.04, -0.10), 0.04, self._analytics, align=TextNode.ARight,
        )
        self._analytics_values = []
        for i, label in enumerate(
            ("Search Time", "Memory Usage", "Expanded Nodes", "Solution Length")
        ):
            z = -0.23 - i * 0.1
            self._text(
                label, (-0.76, z), 0.042, self._analytics,
                fg=(0.62, 0.70, 0.82, 1), align=TextNode.ALeft,
            )
            self._analytics_values.append(self._text(
                "", (-0.04, z), 0.047, self._analytics,
                fg=(1, 1, 1, 1), align=TextNode.ARight, bold=True,
            ))
        self._analytics.hide()

    def _setup_playback_controls(self) -> None:
        """Play / Pause / Step controls, shown once a solution path is loaded."""
        self._playback_frame = DirectFrame(
            parent=self.aspect2d, pos=(0, 0, -0.66), relief=DGG.FLAT,
            frameSize=(-0.70, 0.70, -0.05, 0.16), frameColor=PANEL_COLOR,
        )
        self._playback_label = self._text(
            "", (0, 0.095), 0.038, self._playback_frame,
        )
        make = lambda label, x, w, cmd, args=(): self._make_button(  # noqa: E731
            label, (x, 0, 0), cmd, self._playback_frame, w, 0.05, extra_args=args,
        )
        self._btn_step_back = make("Step Back", -0.45, 6.4, self.playback_step, (-1,))
        self._btn_play = make("Play", -0.14, 3.8, self.playback_play)
        self._btn_pause = make("Pause", 0.10, 4.4, self.playback_pause)
        self._btn_step_fwd = make("Step Forward", 0.44, 8.4, self.playback_step, (1,))
        self._playback_frame.hide()

    HELP_LINES = (
        ("Arrow Keys / WASD", "Roll block"),
        ("Space", "Switch cube (split)"),
        ("R", "Restart level"),
        ("N", "New game"),
        ("Q / E (hold)", "Rotate camera"),
        ("Right Mouse Drag", "Pan camera"),
        ("Mouse Wheel", "Zoom"),
        ("C", "Reset camera"),
        ("F", "Focus on player"),
        ("H", "Show / hide this help"),
        ("Esc", "Level menu / quit"),
    )

    def _setup_help_overlay(self) -> None:
        """Semi-transparent controls tutorial pinned to the top-left corner."""
        self._help_visible = True
        self._help_button = self._make_button(
            "Help [H]", (0.20, 0, -0.08), self.toggle_help, self.a2dTopLeft, 4.6,
            0.05,
        )
        row, top = 0.075, 0.095
        height = top + row * len(self.HELP_LINES) + 0.03
        self._help_frame = DirectFrame(
            parent=self.a2dTopLeft, pos=(0.02, 0, -0.15), relief=DGG.FLAT,
            frameSize=(0, 0.72, -height, 0), frameColor=(0.04, 0.06, 0.10, 0.6),
        )
        self._text(
            "CONTROLS", (0.03, -0.065), 0.045, self._help_frame,
            fg=(0.97, 0.9, 0.66, 1), align=TextNode.ALeft, bold=True,
        )
        for i, (keys, desc) in enumerate(self.HELP_LINES):
            z = -top - row * (i + 0.6)
            self._text(
                keys, (0.03, z), 0.036, self._help_frame,
                fg=(1, 1, 1, 1), align=TextNode.ALeft, bold=True,
            )
            self._text(
                desc, (0.69, z), 0.034, self._help_frame,
                fg=(0.78, 0.84, 0.93, 1), align=TextNode.ARight,
            )

    def toggle_help(self) -> None:
        self._help_visible = not self._help_visible
        if self._help_visible:
            self._help_frame.show()
        else:
            self._help_frame.hide()

    def _bind_keys(self) -> None:
        keys: dict[str, Action] = {
            "w": "UP", "arrow_up": "UP",
            "s": "DOWN", "arrow_down": "DOWN",
            "a": "LEFT", "arrow_left": "LEFT",
            "d": "RIGHT", "arrow_right": "RIGHT",
        }
        for key, action in keys.items():


            self.accept(key, self.try_move, [action])
            self.accept(f"{key}-repeat", self.try_move, [action])


        self.accept("wheel_up", self._zoom_camera, [0.8])
        self.accept("wheel_down", self._zoom_camera, [1.25])
        self.accept("mouse3", self._start_camera_drag)
        self.accept("mouse3-up", self._stop_camera_drag)
        self.accept("c", self._reset_camera)
        self.accept("f", self._focus_player)
        self.accept("space", self.swap_cube)
        self.accept("h", self.toggle_help)
        self.accept("r", self.restart_level)
        self.accept("n", self.new_game)
        self.accept("escape", self._escape_or_quit)
        for key, direction in (("q", 1), ("e", -1)):
            self.accept(key, self._set_camera_spin, [direction])
            self.accept(f"{key}-up", self._set_camera_spin, [0])

    def _refresh_block(self) -> None:
        self._reset_win_sink()
        self._refresh_bridges()
        if self._block_visual is not None:
            self._block_visual.removeNode()
            self._block_visual = None
        for cube in self._cube_visuals:
            cube.removeNode()
        self._cube_visuals.clear()

        if self.state.cubes is None:
            pose = block_to_world(self.state.block)
            self._block_visual = self._place_box(
                pose, (0.98, 0.76, 0.22, 1), "player-block"
            )
        else:
            for i, (row, col) in enumerate(self.state.cubes):
                color = (1, .85, .28, 1) if i == self.state.active_cube else (.77, .49, .20, 1)
                self._cube_visuals.append(self._place_box(
                    BoxPose(float(col), float(-row), 0.46, .92, .92, .92),
                    color, f"cube-{i + 1}",
                ))

        self._update_player_info()

    def _finish_win(self) -> None:
        """Animator callback: the winning roll ended, so redraw the block and sink it."""
        self._refresh_block()
        self._start_win_sink()

    def _start_win_sink(self) -> None:
        """The block stands on the goal: it descends into the hole, hidden below the floor."""
        if not is_goal(self.level, self.state) or self._block_visual is None:
            return
        self._reset_win_sink()
        block = self._block_visual
        # Clip everything under the board surface (z < 0), so the bottom of the block
        # disappears into the goal tile instead of showing it fall through the void.
        plane = self.render.attachNewNode(PlaneNode("goal-clip", Plane(Vec3(0, 0, 1), Point3(0, 0, 0))))
        block.setClipPlane(plane)
        self._sink_clip = plane
        pos = block.getPos()
        self._win_sequence = Sequence(
            Wait(0.1),
            LerpPosInterval(
                block, SINK_SECONDS, (pos.x, pos.y, pos.z - SINK_DROP), blendType="easeInOut",
            ),
            Func(block.hide),
            Func(self._show_win_popup),
            name="win-sink",
        )
        self._win_sequence.start()

    def _reset_win_sink(self) -> None:
        """Stop a running sink and remove its clip plane."""
        if self._win_sequence is not None:
            self._win_sequence.pause()
            self._win_sequence = None
        if self._sink_clip is not None:
            if self._block_visual is not None:
                self._block_visual.clearClipPlane()
            self._sink_clip.removeNode()
            self._sink_clip = None
        if self._win_popup_sequence is not None:
            self._win_popup_sequence.pause()
            self._win_popup_sequence = None
        self._win_popup.hide()

    def _update_player_info(self) -> None:
        """Show the authoritative logical state during visual interpolation."""
        if self.state.cubes is None:
            self._info.setText(
                f"Moves: {self.steps}   |   "
                f"Orientation: {self.state.block.orientation}   |   "
                f"Cell: ({self.state.block.row}, {self.state.block.col})"
            )
        else:
            self._info.setText(
                f"Moves: {self.steps} | SPLIT | Active: {self.state.active_cube + 1} "
                f"| Cubes: {self.state.cubes[0]}, {self.state.cubes[1]}"
            )

    def _cancel_player_animation(self) -> None:
        if self._animator is not None:
            self._animator.cancel()

    def _set_status(self, message: str) -> None:
        self._status.setText(message)

    def swap_cube(self) -> None:
        """Space swaps the active cube only after splitting."""
        self.try_move("SWAP")

    def _setup_win_popup(self) -> None:
        """Congratulation card shown when the block has sunk into the goal."""
        gold = ACTIVE_LEVEL_BORDER
        self._win_popup = DirectFrame(
            parent=self.aspect2d, pos=(0, 0, 0.12), relief=DGG.FLAT,
            frameSize=(-0.86, 0.86, -0.50, 0.50), frameColor=gold,
        )
        card = DirectFrame(
            parent=self._win_popup, relief=DGG.FLAT,
            frameSize=(-0.83, 0.83, -0.47, 0.47), frameColor=PANEL_COLOR,
        )
        DirectFrame(     # accent rule under the title
            parent=card, pos=(0, 0, 0.22), relief=DGG.FLAT,
            frameSize=(-0.45, 0.45, -0.004, 0.004), frameColor=gold,
        )
        self._text(
            "CONGRATULATIONS!", (0, 0.30), 0.105, card,
            fg=(0.99, 0.84, 0.32, 1), bold=True, shadow=(0, 0, 0, 0.75),
        )
        self._win_level_text = self._text(
            "", (0, 0.10), 0.062, card, fg=(0.96, 0.98, 1, 1), bold=True,
        )
        self._win_moves_text = self._text(
            "", (0, -0.01), 0.048, card, fg=(0.62, 0.70, 0.82, 1),
        )
        self._win_again_btn = self._make_button(
            "Play Again", (-0.50, 0, -0.33), self.restart_level, card, 7.4, 0.055,
        )
        self._win_next_btn = self._make_button(
            "Next Level", (0.0, 0, -0.33), self._next_level, card, 7.4, 0.055,
            color=ACTIVE_LEVEL_COLOR,
        )
        self._make_button(
            "Levels", (0.50, 0, -0.33), self.new_game, card, 5.0, 0.055,
        )
        self._win_popup.hide()

    def _show_win_popup(self) -> None:
        """Fill in the result and pop the card in (fade + slight scale-up)."""
        has_next = self._current_level_index + 1 < len(self._level_entries)
        if has_next:
            self._win_next_btn.show()
        else:
            self._win_next_btn.hide()
        moves = "move" if self.steps == 1 else "moves"
        self._win_level_text.setText(f"{self.level.name} solved")
        self._win_moves_text.setText(f"Completed in {self.steps} {moves}")
        popup = self._win_popup
        popup.setScale(0.8)
        popup.setColorScale(1, 1, 1, 0)
        popup.show()
        self._win_popup_sequence = Parallel(
            LerpScaleInterval(popup, 0.35, 1.0, startScale=0.8, blendType="easeOut"),
            LerpColorScaleInterval(popup, 0.35, (1, 1, 1, 1), startColorScale=(1, 1, 1, 0)),
            name="win-popup",
        )
        self._win_popup_sequence.start()

    def _next_level(self) -> None:
        index = self._current_level_index + 1
        if index < len(self._level_entries):
            self._load_level(index)

    def _setup_game_over_panel(self) -> None:
        """Banner shown when the fall animation ends; Restart/New Game stay clickable."""
        self._game_over_panel = DirectFrame(
            parent=self.aspect2d, pos=(0, 0, 0.3), relief=DGG.FLAT,
            frameSize=(-0.75, 0.75, -0.30, 0.25), frameColor=PANEL_COLOR,
        )
        self._text(
            "GAME OVER", (0, 0.04), 0.13, self._game_over_panel,
            fg=(0.95, 0.35, 0.33, 1), bold=True, shadow=(0, 0, 0, 0.75),
        )
        self._text(
            "Click Restart to try again,", (0, -0.1), 0.045, self._game_over_panel,
        )
        self._text(
            "or New Game to choose a level.", (0, -0.19), 0.045, self._game_over_panel,
        )
        self._game_over_panel.hide()

    def _invalid_landing(self, action: SearchAction) -> GameState | None:
        """Where a move onto missing floor would end up, or None if it is just blocked."""
        if action == "SWAP":
            return None
        state = self.state
        if state.cubes is None:
            landing = roll_block(state.block, action)
            if is_legal(self.level, landing, state.bridge_mask):
                return None
            return GameState(landing, state.bridge_mask)
        row, col = state.cubes[state.active_cube]
        dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[action]
        new_pos = (row + dr, col + dc)
        if new_pos == state.cubes[1 - state.active_cube]:
            return None
        if self.level.is_supported(*new_pos, state.bridge_mask):
            return None
        cubes = list(state.cubes)
        cubes[state.active_cube] = new_pos
        return GameState(
            state.block, state.bridge_mask, (cubes[0], cubes[1]), state.active_cube
        )

    def _begin_game_over(self, action: SearchAction, landing: GameState) -> None:
        """Lock input, finish the roll onto the void, then drop the block."""
        self.game_over = True
        self._fall_landing = landing
        self._fall_action = action
        self._clear_solver_ui()
        self._set_status("You fell off the board!")
        if not self._animator.start(self.state, landing, action, on_done=self._start_fall):
            self._start_fall()

    def _start_fall(self) -> None:
        landing = self._fall_landing
        if landing is None or not self.game_over:
            return
        if landing.cubes is None:
            if self._block_visual is not None:
                self._block_visual.removeNode()
            node = self._place_box(
                block_to_world(landing.block), (0.98, 0.76, 0.22, 1), "falling-block"
            )
            self._block_visual = node
        else:
            if len(self._cube_visuals) != 2:
                self._refresh_block()
            row, col = landing.cubes[self.state.active_cube]
            node = self._cube_visuals[self.state.active_cube]
            node.setPos(float(col), float(-row), 0.46)

        pos = node.getPos()
        pitch, roll = {
            "UP": (-FALL_TUMBLE, 0), "DOWN": (FALL_TUMBLE, 0),
            "LEFT": (0, -FALL_TUMBLE), "RIGHT": (0, FALL_TUMBLE),
        }[self._fall_action]
        self._fall_sequence = Sequence(
            Wait(0.12),
            Parallel(
                LerpPosInterval(
                    node, FALL_SECONDS, (pos.x, pos.y, pos.z - FALL_DROP),
                    blendType="easeIn",
                ),
                LerpHprInterval(
                    node, FALL_SECONDS, (FALL_HEADING, pitch, roll), blendType="easeIn",
                ),
            ),
            Func(self._show_game_over),
            name="game-over-fall",
        )
        self._fall_sequence.start()

    def _show_game_over(self) -> None:
        self._game_over_panel.show()
        self._set_status("Game over. Restart or pick a new game.")

    def _reset_game_over(self) -> None:
        if self._fall_sequence is not None:
            self._fall_sequence.pause()
            self._fall_sequence = None
        self.game_over = False
        self._fall_landing = None
        self._game_over_panel.hide()

    def try_move(self, action: SearchAction) -> None:
        if self.game_over or self._solution_running:
            return
        if self._selector_open:
            return
        if self._animator is not None and self._animator.active:

            return
        if is_goal(self.level, self.state):
            self._set_status("Level solved. Press R or New Game to restart.")
            return
        candidate = transition(self.level, self.state, action)
        if candidate is None:
            landing = self._invalid_landing(action)
            if landing is not None:
                self._begin_game_over(action, landing)
                return
            if action == "SWAP":
                self._set_status("Split first to swap cubes.")
            else:
                self._set_status(f"{action}: illegal move.")
            return
        if self._playback_states:
            self._clear_playback()   # a manual move leaves the solution path
        previous_state = self.state
        previous_mask = self.state.bridge_mask
        was_split = self.state.is_split
        self.state = candidate

        if action != "SWAP":
            self.steps += 1
        won = is_goal(self.level, candidate)
        started_animation = (
            self._animator is not None
            and self._animator.start(
                previous_state, candidate, action,
                on_done=self._finish_win if won else None,
            )
        )
        if started_animation:


            self._update_player_info()
        else:
            self._refresh_block()
            if won:
                self._start_win_sink()
        if won:
            self._set_status("")
        elif was_split != self.state.is_split:
            self._set_status("SPLIT! Space to swap cubes." if self.state.is_split else "Cubes merged.")
        elif action == "SWAP":
            self._set_status(f"Active cube: {self.state.active_cube + 1}.")
        elif self.state.bridge_mask != previous_mask:
            changes = ", ".join(
                f"{group.id}={'open' if self.state.bridge_mask & (1 << index) else 'closed'}"
                for index, group in enumerate(self.level.bridges)
                if bool(previous_mask & (1 << index)) != bool(self.state.bridge_mask & (1 << index))
            )
            self._set_status(f"{action}: {changes}")
        else:
            self._set_status(f"Moved {action}.")

    def restart_level(self) -> None:
        self._reset_game_over()
        self._clear_solver_ui()
        self._cancel_player_animation()
        self._hide_level_selector()
        self.state = initial_state(self.level)
        self.steps = 0
        self._refresh_block()
        self._set_status("Restarted level.")

    def new_game(self) -> None:
        if self._selector_open:
            self._hide_level_selector()
        else:
            self._show_level_selector()

    def _setup_level_selector(self) -> None:
        """Build the modal once: dimmed overlay, panel, then one page of levels."""
        self._selector_overlay = DirectFrame(
            frameSize=(-10, 10, -10, 10), frameColor=OVERLAY_COLOR,
            relief=DGG.FLAT, state=DGG.NORMAL, sortOrder=100,
        )
        self._selector_panel = DirectFrame(
            frameSize=(-1.30, 1.30, -0.70, 0.73),
            frameColor=(0.055, 0.085, 0.14, 1), relief=DGG.FLAT, sortOrder=101,
        )
        self._selector_overlay.hide()
        self._selector_panel.hide()
        self._text(
            "CHOOSE LEVEL", (0, 0.5), 0.072, self._selector_panel,
            fg=(1, 1, 1, 1), bold=True,
        )
        self._selector_page_text = self._text(
            "", (0, -0.55), 0.045, self._selector_panel, fg=(0.9, 0.93, 1, 1),
        )
        self._selector_prev = self._make_button(
            "< Prev", (-0.65, 0, -0.57), self._change_selector_page,
            self._selector_panel, 4.4, 0.065, extra_args=(-1,),
        )
        self._selector_next = self._make_button(
            "Next >", (0.65, 0, -0.57), self._change_selector_page,
            self._selector_panel, 4.4, 0.065, extra_args=(1,),
        )
        # Square close button flush with the panel's top-right corner (0.02 inset).
        close_scale, inset = 0.075, 0.02
        half = 1.5 * close_scale / 2
        self._make_button(
            "X", (1.30 - inset - half, 0, 0.73 - inset - half - 0.35 * close_scale),
            self._hide_level_selector, self._selector_panel, 1.5, close_scale,
            color=(0.60, 0.16, 0.19, 1),
        )
        self._populate_selector_page()

    def _populate_selector_page(self) -> None:
        """Eight level buttons per page so large catalogs fit on a 1200x760 UI."""
        for widget in self._selector_widgets:
            widget.destroy()
        self._selector_widgets.clear()
        page_size = 8
        scale = 0.063
        begin = self._selector_page * page_size
        end = min(begin + page_size, len(self._level_entries))
        for offset, index in enumerate(range(begin, end)):
            entry = self._level_entries[index]
            column, row = divmod(offset, 4)
            x = -0.57 if column == 0 else 0.57
            z = 0.31 - row * 0.17
            active = index == self._current_level_index
            if active:
                # Gold border: a slightly larger frame behind the button.
                self._selector_widgets.append(DirectFrame(
                    parent=self._selector_panel, pos=(x, 0, z + 0.35 * scale),
                    frameSize=(-0.565, 0.565, -0.071, 0.071),
                    frameColor=ACTIVE_LEVEL_BORDER, relief=DGG.FLAT,
                ))
            label = f"{entry.name} - {entry.summary}" if entry.summary else entry.name
            self._selector_widgets.append(self._make_button(
                label, (x, 0, z), self._choose_level, self._selector_panel,
                17.5, scale, ACTIVE_LEVEL_COLOR if active else (0.12, 0.35, 0.55, 1),
                extra_args=(index,), text_scale=0.88,
            ))
        count_pages = (len(self._level_entries) + page_size - 1) // page_size
        self._selector_page_text.setText(
            f"Page {self._selector_page + 1}/{count_pages} "
        )
        self._selector_prev["state"] = (
            DGG.NORMAL if self._selector_page > 0 else DGG.DISABLED
        )
        self._selector_next["state"] = (
            DGG.NORMAL if self._selector_page + 1 < count_pages else DGG.DISABLED
        )

    def _change_selector_page(self, direction: int) -> None:
        page_count = (len(self._level_entries) + 7) // 8
        next_page = self._selector_page + direction
        if self._selector_open and 0 <= next_page < page_count:
            self._selector_page = next_page
            self._populate_selector_page()

    def _show_level_selector(self) -> None:
        self._clear_solver_ui()
        if self._selector_panel is None:
            return
        if not self.game_over:   # keep the falling block while a game is over
            self._cancel_player_animation()
            self._refresh_block()
        self._selector_page = max(self._current_level_index, 0) // 8
        self._populate_selector_page()
        self._selector_overlay.show()
        self._selector_panel.show()
        self._selector_open = True
        self._set_status("Choose a level.")

    def _hide_level_selector(self) -> None:
        self._selector_open = False
        if self._selector_panel is not None:
            self._selector_overlay.hide()
            self._selector_panel.hide()

    def _escape_or_quit(self) -> None:
        if self._selector_open:
            self._hide_level_selector()
        else:
            self.userExit()

    def _clear_level_visuals(self) -> None:
        """Remove ALL old level-owned nodes, not just the moving block."""
        for visual in self._tile_visuals:
            visual.removeNode()
        self._tile_visuals.clear()
        for visual in self._bridge_visuals.values():
            visual.removeNode()
        self._bridge_visuals.clear()
        if self._block_visual is not None:
            self._block_visual.removeNode()
            self._block_visual = None
        for visual in self._cube_visuals:
            visual.removeNode()
        self._cube_visuals.clear()

    def _choose_level(self, index: int) -> None:
        """Selector click: change level, only while the selector is open."""
        if not self._selector_open:
            return
        self._load_level(index)

    def _load_level(self, index: int) -> None:
        """Change level in the same ShowBase window and reset every state."""
        entry = self._level_entries[index]
        try:
            new_level = load_level(entry.path)
            new_state = initial_state(new_level)
        except (OSError, ValueError) as exc:
            self._set_status(f"Cannot open level: {exc}")
            return
        self._reset_game_over()
        self._clear_solver_ui()
        self._cancel_player_animation()
        self._clear_level_visuals()
        self.level = new_level
        self._current_level_index = index
        self.state = new_state
        self.steps = 0
        self._setup_camera()
        self._draw_level()
        self._refresh_block()
        self._title.setText(f"BLOXORZ | {self.level.name}")
        if self.level.width >= 35 or self.level.height >= 25:
            self._focus_player()
        self._hide_level_selector()
        self._set_status(f"Level: {self.level.name}")

    def solve_bfs(self) -> None:
        self._run_solver("BFS", bfs)

    def solve_dfs(self) -> None:
        self._run_solver("DFS", dfs)

    def solve_ucs(self) -> None:
        self._run_solver("UCS", ucs)

    def solve_astar(self) -> None:
        self._run_solver("A*", astar)

    def _run_solver(self, name, solver) -> None:
        if (self.game_over or self._solution_running or self._selector_open
                or self._animator.active):
            return
        if is_goal(self.level, self.state):
            self._set_status("Level already solved.")
            return

        self._clear_solver_ui()
        result = solver(self.level, self.state)
        if result is None:
            self._set_status(f"{name}: not implemented.")
            return
        if not isinstance(result, SearchResult):
            self._set_status(f"{name}: expected SearchResult.")
            return

        self._show_analytics(name, result)
        if result.path is None:
            self._set_status(f"{name}: no solution.")
            return
        if not result.path:
            self._set_status(f"{name}: empty path.")
            return

        # Replay the path through the engine once; the states double as the
        # timeline that Step Back / Step Forward jump between.
        states = [self.state]
        moves = [0]
        for action in result.path:
            if action not in ("UP", "DOWN", "LEFT", "RIGHT", "SWAP"):
                self._set_status(f"{name}: invalid action.")
                return
            next_state = transition(self.level, states[-1], action)
            if next_state is None or is_goal(self.level, states[-1]):
                self._set_status(f"{name}: invalid path.")
                return
            states.append(next_state)
            moves.append(moves[-1] + (action != "SWAP"))
        if not is_goal(self.level, states[-1]):
            self._set_status(f"{name}: path does not reach goal.")
            return

        self._playback_name = name
        self._playback_path = list(result.path)
        self._playback_states = states
        self._playback_moves = moves
        self._playback_index = 0
        self._playback_base_steps = self.steps
        self._playback_frame.show()
        self.playback_play()

    def _show_analytics(self, name: str, result: SearchResult) -> None:
        memory_kb = result.peak_memory / 1024
        length = result.solution_length
        self._analytics_title.setText(f"{name} Results")
        self._analytics_outcome.setText("Solved" if result.path is not None else "No solution")
        for widget, value in zip(self._analytics_values, (
            f"{result.search_time * 1000:,.2f} ms",
            f"{memory_kb:,.1f} KB",
            f"{result.expanded_nodes:,}",
            "--" if length is None else f"{length:,} moves",
        )):
            widget.setText(value)
        self._analytics.show()

    def _update_playback_ui(self) -> None:
        total = len(self._playback_path)
        self._playback_label.setText(
            f"{self._playback_name} playback  -  step {self._playback_index} / {total}"
        )
        playing = self._solution_running
        for button, enabled in (
            (self._btn_play, not playing),
            (self._btn_pause, playing),
            (self._btn_step_back, self._playback_index > 0),
            (self._btn_step_fwd, self._playback_index < total),
        ):
            button["state"] = DGG.NORMAL if enabled else DGG.DISABLED

    def _playback_goto(self, index: int, animate: bool = False) -> None:
        """Jump the game to timeline position `index` (0 = where the solver started)."""
        index = max(0, min(index, len(self._playback_path)))
        previous = self._playback_index
        if index == previous:
            return
        self._cancel_player_animation()
        before = self._playback_states[previous]
        self._playback_index = index
        self.state = self._playback_states[index]
        self.steps = self._playback_base_steps + self._playback_moves[index]
        forward = index == previous + 1
        won = is_goal(self.level, self.state)
        if animate and forward and self._animator.start(
            before, self.state, self._playback_path[previous],
            on_done=self._finish_win if won else None,
        ):
            self._update_player_info()
        else:
            self._refresh_block()
            if won:
                self._start_win_sink()
        if won:
            self._set_status("")
        elif index == 0:
            self._set_status(f"{self._playback_name}: back at the start.")
        else:
            self._set_status(
                f"{self._playback_name}: step {index}/{len(self._playback_path)} "
                f"({self._playback_path[index - 1]})"
            )
        self._update_playback_ui()

    def playback_play(self) -> None:
        if not self._playback_path or self._solution_running:
            return
        if self._playback_index >= len(self._playback_path):
            self._playback_goto(0)   # replay from the beginning
        self._solution_running = True
        self.taskMgr.add(self._play_solution, "solution-playback")
        self._update_playback_ui()

    def playback_pause(self) -> None:
        self._stop_solution()

    def playback_step(self, direction: int) -> None:
        if not self._playback_path:
            return
        self._stop_solution()
        self._playback_goto(self._playback_index + direction, animate=direction > 0)

    def _play_solution(self, task):
        if self._animator.active:
            return task.cont
        if self._playback_index >= len(self._playback_path):
            self._stop_solution()
            return task.done
        self._playback_goto(self._playback_index + 1, animate=True)
        return task.cont

    def _stop_solution(self) -> None:
        """Pause playback; the loaded path and position are kept."""
        self._solution_running = False
        self.taskMgr.remove("solution-playback")
        if self._playback_path:
            self._update_playback_ui()

    def _clear_playback(self) -> None:
        self._stop_solution()
        self._playback_path = []
        self._playback_states = []
        self._playback_moves = []
        self._playback_index = 0
        self._playback_frame.hide()

    def _clear_solver_ui(self) -> None:
        """Forget the previous solver run: playback controls and metrics card."""
        self._clear_playback()
        self._analytics.hide()

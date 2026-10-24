"""Generate new, validated Bloxorz test levels as fixed JSON files.

Usage (from the repo root):
    python experiments/generate_levels.py --group 1 --count 5 --size medium --seed 1

Each run is deterministic for a given (group, size, seed, existing levels). Every
candidate map is rejected and resampled unless it loads, falls into the requested
group, is solved by BFS within the limits below and is long enough. Accepted levels
are written to Source/levels/testN.json, inserted into campaign.json before Test All
and recorded in experiments/levels_manifest.csv.
"""

import argparse
import csv
import json
import random
import re
import sys
import tempfile
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
if (EXP_DIR.parent / "models").exists():
    SOURCE_DIR = EXP_DIR.parent
elif (EXP_DIR.parent / "Source" / "models").exists():
    SOURCE_DIR = EXP_DIR.parent / "Source"
else:
    raise RuntimeError("Cannot find the Source directory next to 'experiments'")

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from models.catalog import load_catalog
from models.level import load_level
from engine.transitions import initial_state, transition
from solvers.bfs import bfs
from solvers.ucs import ucs
from solvers.astar import astar
from run_experiments import classify_level  # main() is guarded; importing runs nothing

LEVELS_DIR = SOURCE_DIR / "levels"
CAMPAIGN = LEVELS_DIR / "campaign.json"
LAST_LEVEL = "test_all.json"  # "Test All" uses every mechanic and stays last in the menu
MANIFEST = EXP_DIR / "levels_manifest.csv"
MANIFEST_FIELDS = ["file", "name", "group", "size", "seed", "bfs_moves", "bfs_expanded"]

SIZES = {"small": (8, 12), "medium": (12, 25), "large": (16, 40)}  # rows x cols
MIN_MOVES = {"small": 10, "medium": 18, "large": 25}
WAYPOINTS = {"small": (1, 2), "medium": (2, 3), "large": (3, 5)}
FRAGILE_COUNT = {"small": (1, 2), "medium": (2, 4), "large": (3, 6)}
BRIDGE_CUTS = {"small": (1, 1), "medium": (1, 1), "large": (1, 2)}
MAX_EXPANDED = 300_000
MAX_SECONDS = 10.0
MAX_ATTEMPTS = 20_000
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class HeuristicMismatch(Exception):
    """UCS and A* disagree on the optimal number of Moves."""


# ---------------------------------------------------------------- map carving

def _carve_walk(rng, grid, a, b):
    """Biased random walk from a to b inside the 1-cell void border."""
    h, w = len(grid), len(grid[0])
    r, c = a
    grid[r][c] = "."
    cells = [(r, c)]
    while (r, c) != b:
        if rng.random() < 0.7:
            options = []
            if r != b[0]:
                options.append((1 if b[0] > r else -1, 0))
            if c != b[1]:
                options.append((0, 1 if b[1] > c else -1))
            dr, dc = rng.choice(options)
        else:
            dr, dc = rng.choice(DIRS)
        if 1 <= r + dr <= h - 2 and 1 <= c + dc <= w - 2:
            r, c = r + dr, c + dc
            grid[r][c] = "."
            cells.append((r, c))
    return cells


def _carve_map(rng, size):
    """Walk S -> waypoints -> G (left to right), widen some cells into rooms."""
    h, w = SIZES[size]
    grid = [["#"] * w for _ in range(h)]
    quarter = max(2, w // 4)
    start = (rng.randint(1, h - 2), rng.randint(1, quarter))
    goal = (rng.randint(1, h - 2), rng.randint(w - 1 - quarter, w - 2))
    lo, hi = WAYPOINTS[size]
    middle = sorted(
        ((rng.randint(1, h - 2), rng.randint(quarter, w - 1 - quarter))
         for _ in range(rng.randint(lo, hi))),
        key=lambda p: p[1],
    )
    points = [start, *middle, goal]
    walk = []
    for a, b in zip(points, points[1:]):
        walk += _carve_walk(rng, grid, a, b)
    for r, c in walk:
        if rng.random() < 0.25:
            rh, rw = rng.randint(1, 3), rng.randint(1, 3)
            r0, c0 = r - rng.randint(0, rh - 1), c - rng.randint(0, rw - 1)
            for rr in range(max(1, r0), min(h - 1, r0 + rh)):
                for cc in range(max(1, c0), min(w - 1, c0 + rw)):
                    grid[rr][cc] = "."
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"
    return grid, start, goal, walk


def _reachable(grid, start, passable):
    """4-connected flood fill over cells (r, c) for which passable(r, c) holds."""
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and (nr, nc) not in seen and passable(nr, nc)):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _add_bridges(rng, grid, size, start):
    """Void whole columns, leaving a closed bridge as the only crossing; add switches.

    Switch i is placed where the block can get before crossing bridge i, so every
    bridge is controlled by a reachable switch. Returns (bridges, switches) or None.
    """
    h, w = len(grid), len(grid[0])
    quarter = max(2, w // 4)
    lo, hi = BRIDGE_CUTS[size]
    columns = list(range(quarter + 1, w - quarter - 1))
    rng.shuffle(columns)
    cuts = []
    for c in columns:
        if all(abs(c - other) >= 3 for other in cuts):
            cuts.append(c)
        if len(cuts) == rng.randint(lo, hi):
            break
    cuts.sort()

    bridges = []
    for index, c in enumerate(cuts):
        crossings = [r for r in range(1, h - 1)
                     if grid[r][c - 1] in ".F" and grid[r][c] == "." and grid[r][c + 1] in ".F"]
        if not crossings:
            return None
        r = rng.choice(crossings)
        cells = [(r, c)]
        for nr in (r - 1, r + 1):
            if nr in crossings and rng.random() < 0.4:
                cells.append((nr, c))
        for rr in range(h):
            grid[rr][c] = "#"
        for rr, cc in cells:
            grid[rr][cc] = "B"
        bridges.append({"id": chr(ord("A") + index),
                        "cells": [[rr, cc] for rr, cc in sorted(cells)],
                        "initially_open": False})

    switches = []
    crossable: set[tuple[int, int]] = set()  # cells of bridges already given a switch
    for bridge in bridges:
        region = _reachable(grid, start, lambda r, c: grid[r][c] in ".FoX" or (r, c) in crossable)
        candidates = sorted(p for p in region if grid[p[0]][p[1]] == ".")
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        kind = rng.choice(("soft", "heavy"))
        grid[r][c] = "o" if kind == "soft" else "X"
        switch = {"position": [r, c], "type": kind, "bridges": [bridge["id"]]}
        if rng.random() < 0.6:
            switch["effect"] = "permanent"
            switch["set_open"] = True
        else:
            switch["effect"] = "toggle"
        switches.append(switch)
        crossable.update(tuple(cell) for cell in bridge["cells"])
    return bridges, switches


def _add_fragile(rng, grid, size, walk):
    """Turn a few floor cells, mostly on the carved walk, into fragile tiles."""
    floor = sorted({(r, c) for r, c in walk if grid[r][c] == "."})
    if not floor:
        return
    for r, c in rng.sample(floor, min(len(floor), rng.randint(*FRAGILE_COUNT[size]))):
        grid[r][c] = "F"


def _make_candidate(rng, group, size):
    grid, start, _, walk = _carve_map(rng, size)
    data = {}
    if group in (3, 4):
        added = _add_bridges(rng, grid, size, start)
        if added is None:
            return None
        data["bridges"], data["switches"] = added
    if group in (2, 4):
        _add_fragile(rng, grid, size, walk)
    data["map"] = ["".join(row) for row in grid]
    return data


# ---------------------------------------------------------------- validation

def _path_states(level, path):
    state = initial_state(level)
    states = [state]
    for action in path:
        state = transition(level, state, action)
        states.append(state)
    return states


def _evaluate(data, group, size, tmp_dir):
    """Return (level, bfs_result) if the candidate passes every check, else None."""
    tmp = Path(tmp_dir) / "candidate.json"
    tmp.write_text(json.dumps({"name": "candidate", **data}), encoding="utf-8")
    try:
        level = load_level(tmp)
        start = initial_state(level)
    except ValueError:
        return None
    if not classify_level(level).startswith(f"Group {group}:"):
        return None
    result = bfs(level, start)
    if (result is None or result.path is None or result.expanded_nodes > MAX_EXPANDED
            or result.search_time > MAX_SECONDS or result.solution_length < MIN_MOVES[size]):
        return None
    states = _path_states(level, result.path)
    if group in (3, 4) and all(s.bridge_mask == start.bridge_mask for s in states):
        return None
    if group in (2, 4) and not any(
        level.tile_at(r, c) == "F" for s in states for r, c in s.occupied_cells()
    ):
        return None
    return level, result


def _check_ucs_astar(level):
    start = initial_state(level)
    u, a = ucs(level, start), astar(level, start)
    if u is None or a is None or u.solution_length != a.solution_length:
        raise HeuristicMismatch(
            f"UCS Moves={u and u.solution_length}, A* Moves={a and a.solution_length}"
        )


# ---------------------------------------------------------------- file output

def _next_index():
    numbers = [int(m.group(1)) for p in LEVELS_DIR.glob("test*.json")
               if (m := re.fullmatch(r"test(\d+)\.json", p.name))]
    return max(numbers, default=0) + 1


def _existing_maps():
    maps = set()
    for p in LEVELS_DIR.glob("*.json"):
        if p.name != CAMPAIGN.name:
            maps.add(tuple(json.loads(p.read_text(encoding="utf-8"))["map"]))
    return maps


def _write_json(path, data):
    # Existing level files use 2-space indentation and CRLF line endings.
    with path.open("w", encoding="utf-8", newline="\r\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _append_campaign(filename, path=CAMPAIGN):
    """Add a level to the campaign, keeping the final "Test All" level last."""
    campaign = json.loads(path.read_text(encoding="utf-8"))
    levels = campaign["levels"]
    files = [entry.get("file") for entry in levels]
    position = files.index(LAST_LEVEL) if LAST_LEVEL in files else len(levels)
    levels.insert(position, {"file": filename})
    _write_json(path, campaign)


def _append_manifest(row):
    is_new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--size", choices=tuple(SIZES), default="medium")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=None,
                        help="first N for testN.json (default: highest existing + 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate and validate only; write nothing")
    args = parser.parse_args()

    index = args.start_index if args.start_index is not None else _next_index()
    rng = random.Random(args.seed)
    seen_maps = _existing_maps()
    written = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for _ in range(args.count):
            filename = f"test{index}.json"
            target = LEVELS_DIR / filename
            if target.exists():
                sys.exit(f"{filename} already exists; refusing to overwrite an existing level")
            for attempt in range(1, MAX_ATTEMPTS + 1):
                data = _make_candidate(rng, args.group, args.size)
                if data is None or tuple(data["map"]) in seen_maps:
                    continue
                accepted = _evaluate(data, args.group, args.size, tmp_dir)
                if accepted is not None:
                    break
            else:
                sys.exit(f"No valid level found for {filename} after {MAX_ATTEMPTS} attempts")
            level, result = accepted
            try:
                _check_ucs_astar(level)
            except HeuristicMismatch as exc:
                print(f"STOP: {filename} candidate: {exc}. Possible heuristic bug; "
                      f"level not written. Map:\n" + "\n".join(data["map"]))
                sys.exit(2)

            seen_maps.add(tuple(data["map"]))
            name = f"Test {index}"
            row = {"file": filename, "name": name, "group": args.group, "size": args.size,
                   "seed": args.seed, "bfs_moves": result.solution_length,
                   "bfs_expanded": result.expanded_nodes}
            if not args.dry_run:
                _write_json(target, {"name": name, **data})
                _append_campaign(filename)
                _append_manifest(row)
            written.append(row)
            print(f"{filename}: group {args.group}, {args.size}, BFS Moves="
                  f"{result.solution_length}, expanded={result.expanded_nodes}, "
                  f"attempts={attempt}")
            if args.dry_run:
                print("\n".join(data["map"]))
            index += 1

    entries = load_catalog(CAMPAIGN)
    print(f"campaign.json OK: {len(entries)} levels validate")


if __name__ == "__main__":
    main()

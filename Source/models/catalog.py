"""Load the ordered list of playable Bloxorz levels."""

from dataclasses import dataclass
import json
from pathlib import Path

from models.level import Level, load_level


@dataclass(frozen=True, slots=True)
class LevelEntry:
    """One playable level, its file path and a short difficulty label."""

    path: Path
    name: str
    summary: str = ""


def describe_level(level: Level) -> str:
    """Name the mechanics a level uses, read from its stage data."""
    mechanics: list[str] = []
    if any("F" in row for row in level.rows):
        mechanics.append("Fragile Tiles")
    if level.switches or level.bridges:
        mechanics.append("Switches & Bridges")
    if level.split_switches:
        mechanics.append("Split Block")
    if not mechanics:
        return "Basic"
    return mechanics[0] if len(mechanics) == 1 else "Combined"


def load_catalog(path: str | Path) -> tuple[LevelEntry, ...]:
    'Read a curated catalog, validate every level, and preserve list order.'
    catalog_path = Path(path).resolve()
    with catalog_path.open(encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, dict) or not isinstance(content.get("levels"), list):
        raise ValueError("Catalog must be a JSON object containing a 'levels' list")
    if not content["levels"]:
        raise ValueError("The level catalog must not be empty")
    base = catalog_path.parent
    entries: list[LevelEntry] = []
    used_paths: set[Path] = set()
    for index, spec in enumerate(content["levels"]):
        if not isinstance(spec, dict) or not isinstance(spec.get("file"), str):
            raise ValueError(f"Catalog entry {index} needs a filename")
        filename = spec["file"]
        if Path(filename).name != filename or not filename.endswith(".json") or filename == 'catalog.json':
            raise ValueError(f"Catalog entry {index}: invalid filename")
        target = (base / filename).resolve()
        if target.parent != base or target in used_paths:
            raise ValueError(f"Catalog entry {index}: repeated/unsafe path")
        level = load_level(target)
        entries.append(LevelEntry(target, level.name, describe_level(level)))
        used_paths.add(target)
    return tuple(entries)

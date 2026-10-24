from pathlib import Path

from gui.app import BloxorzApp
from models.catalog import load_catalog
from models.level import load_level

def main() -> None:
    levels_dir = Path(__file__).resolve().parent / "levels"
    entries = load_catalog(levels_dir / "campaign.json")
    game = BloxorzApp(load_level(entries[0].path), entries)
    game.run()


if __name__ == "__main__":
    main()

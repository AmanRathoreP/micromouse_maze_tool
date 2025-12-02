"""
Convert Micromouse Maze Generator JSON export -> JPEG.
Requires Pillow:  pip install pillow
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from PIL import Image, ImageDraw

MARGIN = 60  # matches the Blazor export
DEFAULT_CELL_SIZE = 40
DEFAULT_WALL_THICKNESS = 3


def load_maze(path: Path) -> Dict[str, Any]:
    # exported files include a UTF-8 BOM, so use utf-8-sig
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pick(data: Dict[str, Any], *keys: str, default: Any = None, allow_default: bool = False) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    if allow_default:
        return default
    raise KeyError(f"Missing keys {keys} in {data}")


class MazeAdapter:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    def rows(self) -> int:
        raise NotImplementedError

    def columns(self) -> int:
        raise NotImplementedError

    def cell_size(self) -> int:
        return DEFAULT_CELL_SIZE

    def wall_thickness(self) -> int:
        return DEFAULT_WALL_THICKNESS

    def show_highlights(self) -> bool:
        return False

    def cells(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError


class LegacyMazeAdapter(MazeAdapter):
    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__(data)
        self.grid = data["grid"]
        self.preview = data["preview"]

    def rows(self) -> int:
        return pick(self.grid, "rows", "Rows")

    def columns(self) -> int:
        return pick(self.grid, "columns", "Columns")

    def cell_size(self) -> int:
        return pick(self.preview, "cellSize", "CellSize", default=DEFAULT_CELL_SIZE, allow_default=True)

    def wall_thickness(self) -> int:
        return pick(
            self.preview,
            "wallThickness",
            "WallThickness",
            default=DEFAULT_WALL_THICKNESS,
            allow_default=True,
        )

    def show_highlights(self) -> bool:
        return bool(self.preview.get("showStartEndCells") or self.preview.get("ShowStartEndCells", True))

    def cells(self) -> Iterable[Dict[str, Any]]:
        return self.grid["cells"]


class ExportedMazeAdapter(MazeAdapter):
    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__(data)
        self.grid = data["grid"]
        self.metadata = data.get("metadata", {})

    def rows(self) -> int:
        return pick(self.grid, "rows", "Rows")

    def columns(self) -> int:
        return pick(self.grid, "columns", "Columns")

    def cell_size(self) -> int:
        return pick(
            self.metadata,
            "cellSize",
            "CellSize",
            default=DEFAULT_CELL_SIZE,
            allow_default=True,
        )

    def wall_thickness(self) -> int:
        return pick(
            self.metadata,
            "wallThickness",
            "WallThickness",
            default=DEFAULT_WALL_THICKNESS,
            allow_default=True,
        )

    def show_highlights(self) -> bool:
        return bool(self.metadata.get("showStartEndCells") or self.metadata.get("ShowStartEndCells", False))

    def cells(self) -> Iterable[Dict[str, Any]]:
        return self.grid["cells"]


def get_adapter(maze: Dict[str, Any]) -> MazeAdapter:
    if "preview" in maze:
        return LegacyMazeAdapter(maze)
    return ExportedMazeAdapter(maze)


def draw_maze(maze: Dict[str, Any], output: Path) -> None:
    adapter = get_adapter(maze)
    rows = adapter.rows()
    cols = adapter.columns()
    cell_size = adapter.cell_size()
    wall_thickness = adapter.wall_thickness()
    show_highlights = adapter.show_highlights()

    width = cols * cell_size + MARGIN * 2
    height = rows * cell_size + MARGIN * 2

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    def cell_origin(cell: Dict[str, Any]) -> Tuple[int, int]:
        col = pick(cell, "column", "Column") - 1  # JSON is 1-based
        row_bottom_origin = pick(cell, "row", "Row")
        ui_row = rows - row_bottom_origin  # convert back to top-left origin
        x = MARGIN + col * cell_size
        y = MARGIN + ui_row * cell_size
        return x, y

    cells = list(adapter.cells())

    if show_highlights:
        for cell in cells:
            x, y = cell_origin(cell)
            cell_type = pick(cell, "cellType", "CellType")
            color = "#90EE90" if cell_type == "start" else "#FFB6C1" if cell_type == "end" else None
            if color:
                draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color)

    for cell in cells:
        x, y = cell_origin(cell)
        walls = pick(cell, "walls", "Walls")
        if pick(walls, "HasTopWall", "hasTopWall"):
            draw.line([x, y, x + cell_size, y], fill="black", width=wall_thickness)
        if pick(walls, "HasRightWall", "hasRightWall"):
            draw.line([x + cell_size, y, x + cell_size, y + cell_size], fill="black", width=wall_thickness)
        if pick(walls, "HasBottomWall", "hasBottomWall"):
            draw.line([x, y + cell_size, x + cell_size, y + cell_size], fill="black", width=wall_thickness)
        if pick(walls, "HasLeftWall", "hasLeftWall"):
            draw.line([x, y, x, y + cell_size], fill="black", width=wall_thickness)

    img.save(output, format="JPEG", quality=95)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    print("generating image...")
    maze_json = Path(r"C:\Users\amanr\Downloads\MazeGenerator_20251201_103045\maze.json")
    output_file = Path("maze_from_json.jpg")
    draw_maze(load_maze(maze_json), output_file)
    print("done")
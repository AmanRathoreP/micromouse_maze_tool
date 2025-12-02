"""Utilities to load Micromouse .maz files and export richer formats.

The main entry point is :class:`MicromouseMaze`. It knows how to:
* read a binary ``.maz`` file (256 bytes, column-major, NESW bit flags)
* render that maze to a JPEG preview using Pillow
* emit a Pydantic-backed JSON document that mirrors ``maze.json``

Example
-------

```python
from pathlib import Path
from maz_to_json import MicromouseMaze

maze = MicromouseMaze.from_file("mazefiles/binary/empty.maz")
maze.save_jpeg(Path("empty.jpg"))
maze.save_json(Path("empty.json"))
```
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, HttpUrl
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITHUB_BASE_URL = "https://github.com/AmanRathoreP/micromouse_maze_tool/blob/master/"
CUSTOM_START_END_MAZES = {
	"beam94a": {
		"start_cells": [(1, 1)],
		"end_cells": [(9, 7), (9, 8), (8, 7), (8, 8)],
	},
}

MAZES_TO_IGNORE = [
			"5x5_test1",
			"5x5_test2",
			"5x5_test3",
			"5x5_test4",
			"11simple",
			"111",
			# "beam94a",  # edit this maze manually
			"empty",
			"map-y77",
			"minimaze",
			"maze-train-10x5",
			"maze-train-10x5-a",
			"maze-train-10x5-q",
			"maze-train-10x5-z",	
		]

class HighlightCell(BaseModel):
	Row: int
	Column: int
	index: int


class WallState(BaseModel):
	HasTopWall: bool
	HasRightWall: bool
	HasBottomWall: bool
	HasLeftWall: bool


class CellModel(BaseModel):
	Row: int
	Column: int
	CellType: str
	walls: WallState


class GridModel(BaseModel):
	Rows: int
	Columns: int
	cells: List[CellModel]


class MetadataModel(BaseModel):
	Rows: int
	Columns: int
	Seed: Optional[int] = None
	Algorithm: str = "Unknown"
	generatedAtUtc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	variant: str = "custom"
	AppVersion: str = "1.0.0"
	DeveloperName: str = "Aman Rathore"
	DeveloperWebsite: Optional[HttpUrl] = Field(default="https://amanr.me")
	ToolWebsite: HttpUrl = Field(default="https://mazegenerator.amanr.me")
	ToolRepository: HttpUrl = Field(default="https://github.com/AmanRathoreP/micromouse_maze_tool")
	MazeRepository: Optional[HttpUrl] = Field(
		default="https://github.com/AmanRathoreP/micromouse_maze_tool/tree/master/mazefiles/binary"
	)
	SourceMazeFile: str
	formats: List[str] = Field(default_factory=lambda: ["JPEG", "JSON"])
	cellSize: int = 40
	wallThickness: int = 3
	ShowStartEndCells: bool = True
	startCells: List[HighlightCell] = Field(default_factory=list)
	endCells: List[HighlightCell] = Field(default_factory=list)


class MazeExportModel(BaseModel):
	metadata: MetadataModel
	grid: GridModel


@dataclass
class MicromouseMaze:
	"""Represents a single micromouse maze encoded as a 16x16 ``.maz`` file."""

	cells: MutableSequence[int]
	source_path: Path

	WIDTH: int = 16
	HEIGHT: int = 16
	DEFAULT_CELL_SIZE: int = 40
	DEFAULT_WALL_THICKNESS: int = 3

	NORTH: int = 0x01
	EAST: int = 0x02
	SOUTH: int = 0x04
	WEST: int = 0x08

	@classmethod
	def from_file(cls, path: Path | str) -> "MicromouseMaze":
		data_path = Path(path)
		blob = data_path.read_bytes()
		expected = cls.WIDTH * cls.HEIGHT
		if len(blob) != expected:
			raise ValueError(
				f"Binary .maz files must be {expected} bytes; got {len(blob)} from {data_path}."
			)
		return cls(list(blob), data_path)

	def save_jpeg(
		self,
		destination: Path | str,
		*,
		cell_size: int = 40,
		wall_thickness: int = 3,
		margin: int = 12,
		start_cells: Optional[Sequence[Tuple[int, int]]] = None,
		end_cells: Optional[Sequence[Tuple[int, int]]] = None,
	) -> Path:
		"""Render the maze as a JPEG image using Pillow."""

		try:
			from PIL import Image, ImageDraw
		except ImportError as exc:  # pragma: no cover - import guard
			raise RuntimeError("Pillow is required for JPEG export. Install 'pillow'.") from exc

		width = self.WIDTH * cell_size + margin * 2
		height = self.HEIGHT * cell_size + margin * 2
		img = Image.new("RGB", (width, height), color=(255, 255, 255))
		draw = ImageDraw.Draw(img)

		def cell_bounds(display_row: int, column: int) -> Tuple[int, int, int, int]:
			top = margin + display_row * cell_size
			left = margin + column * cell_size
			return left, top, left + cell_size, top + cell_size

		start_cells = list(start_cells or self._default_start_cells())
		end_cells = list(end_cells or self._default_goal_cells())

		self._fill_cells(draw, start_cells, cell_size, margin, fill="#90EE90")
		self._fill_cells(draw, end_cells, cell_size, margin, fill="#FFB6C1")

		for column in range(self.WIDTH):
			for internal_row in range(self.HEIGHT - 1, -1, -1):
				display_row = self._display_row(internal_row)
				bounds = cell_bounds(display_row, column)
				byte = self._cell_byte(column, internal_row)
				self._draw_walls(draw, bounds, byte, wall_thickness)

		destination_path = Path(destination)
		destination_path.parent.mkdir(parents=True, exist_ok=True)
		img.save(destination_path, format="JPEG", quality=95)
		return destination_path

	def save_json(
		self,
		destination: Path | str,
		*,
		metadata_overrides: Optional[Mapping[str, object]] = None,
		start_cells: Optional[Sequence[Tuple[int, int]]] = None,
		end_cells: Optional[Sequence[Tuple[int, int]]] = None,
	) -> Path:
		"""Serialize this maze to the structured Pydantic-backed JSON."""

		model = self.to_export_model(
			metadata_overrides=metadata_overrides,
			start_cells=start_cells,
			end_cells=end_cells,
		)
		json_text = (
			model.model_dump_json(indent=2)
			if hasattr(model, "model_dump_json")
			else model.json(indent=2)
		)
		destination_path = Path(destination)
		destination_path.parent.mkdir(parents=True, exist_ok=True)
		destination_path.write_text(json_text, encoding="utf-8")
		return destination_path

	def to_export_model(
		self,
		*,
		metadata_overrides: Optional[Mapping[str, object]] = None,
		start_cells: Optional[Sequence[Tuple[int, int]]] = None,
		end_cells: Optional[Sequence[Tuple[int, int]]] = None,
	) -> MazeExportModel:
		start_cells = list(start_cells or self._default_start_cells())
		end_cells = list(end_cells or self._default_goal_cells())

		start_highlights = [self._highlight_cell(r, c) for r, c in start_cells]
		end_highlights = [self._highlight_cell(r, c) for r, c in end_cells]

		metadata_values = dict(
			Rows=self.HEIGHT,
			Columns=self.WIDTH,
			SourceMazeFile=self._source_url(),
			cellSize=self.DEFAULT_CELL_SIZE,
			wallThickness=self.DEFAULT_WALL_THICKNESS,
			ShowStartEndCells=True,
			startCells=start_highlights,
			endCells=end_highlights,
		)
		if metadata_overrides:
			metadata_values.update(metadata_overrides)

		metadata = MetadataModel(**metadata_values)
		grid = GridModel(
			Rows=self.HEIGHT,
			Columns=self.WIDTH,
			cells=list(self._iter_cells(start_cells, end_cells)),
		)
		return MazeExportModel(metadata=metadata, grid=grid)

	# ------------------------------------------------------------------
	# Internal helpers
	def _source_url(self) -> str:
		resolved = self.source_path.resolve()
		try:
			relative = resolved.relative_to(PROJECT_ROOT)
		except ValueError:
			return str(resolved)
		return f"{GITHUB_BASE_URL}{relative.as_posix()}"

	def _cell_type(
		self,
		row: int,
		column: int,
		start_set: set[Tuple[int, int]],
		end_set: set[Tuple[int, int]],
	) -> str:
		if (row, column) in start_set:
			return "start"
		if (row, column) in end_set:
			return "end"
		return "normal"

	def _highlight_cell(self, row: int, column: int) -> HighlightCell:
		index = (row - 1) * self.WIDTH + column
		return HighlightCell(Row=row, Column=column, index=index)

	def _default_start_cells(self) -> List[Tuple[int, int]]:
		return [(1, 1)]  # bottom-left cell

	def _default_goal_cells(self) -> List[Tuple[int, int]]:
		mid_low = self.HEIGHT // 2
		mid_high = mid_low + 1
		return [
			(mid_low, mid_low),
			(mid_low, mid_high),
			(mid_high, mid_low),
			(mid_high, mid_high),
		]

	def _iter_cells(
		self,
		start_cells: Sequence[Tuple[int, int]],
		end_cells: Sequence[Tuple[int, int]],
	) -> Iterable[CellModel]:
		start_set = {(r, c) for r, c in start_cells}
		end_set = {(r, c) for r, c in end_cells}

		for column in range(self.WIDTH):
			for internal_row in range(self.HEIGHT - 1, -1, -1):
				byte = self._cell_byte(column, internal_row)
				walls = WallState(
					HasTopWall=bool(byte & self.NORTH),
					HasRightWall=bool(byte & self.EAST),
					HasBottomWall=bool(byte & self.SOUTH),
					HasLeftWall=bool(byte & self.WEST),
				)
				row_number = internal_row + 1  # bottom row == 1
				column_number = column + 1  # leftmost column == 1
				cell_type = self._cell_type(row_number, column_number, start_set, end_set)
				yield CellModel(Row=row_number, Column=column_number, CellType=cell_type, walls=walls)

	def _cell_byte(self, column: int, internal_row: int) -> int:
		offset = column * self.HEIGHT + internal_row
		return self.cells[offset]

	def _display_row(self, internal_row: int) -> int:
		return self.HEIGHT - 1 - internal_row

	def _draw_walls(self, draw, bounds, byte: int, thickness: int):  # type: ignore[no-untyped-def]
		left, top, right, bottom = bounds
		for has_wall, segment in (
			(byte & self.NORTH, ((left, top), (right, top))),
			(byte & self.EAST, ((right, top), (right, bottom))),
			(byte & self.SOUTH, ((left, bottom), (right, bottom))),
			(byte & self.WEST, ((left, top), (left, bottom))),
		):
			if has_wall:
				draw.line(segment, fill=(0, 0, 0), width=thickness)

	def _fill_cells(
		self,
		draw,
		cells: Sequence[Tuple[int, int]],
		cell_size: int,
		margin: int,
		*,
		fill: str,
	):  # type: ignore[no-untyped-def]
		for row_number, column_number in cells:
			ui_row = self.HEIGHT - row_number
			x = margin + (column_number - 1) * cell_size
			y = margin + ui_row * cell_size
			draw.rectangle([x, y, x + cell_size, y + cell_size], fill=fill)


	@staticmethod
	def export_all(binary_dir: Path, output_dir: Path) -> List[Tuple[Path, str]]:
		files = sorted(p for p in binary_dir.glob("*.maz") if p.is_file())
		output_dir.mkdir(parents=True, exist_ok=True)
		errors: List[Tuple[Path, str]] = []
		
		for maz_path in tqdm(files, desc="Exporting mazes", unit="maze"):
			if maz_path.stem.lower() in MAZES_TO_IGNORE:
				continue
			json_path = output_dir / f"{maz_path.stem}.json"
			jpeg_path = output_dir / f"{maz_path.stem}.jpg"
			try:
				maze = MicromouseMaze.from_file(maz_path)
				config = CUSTOM_START_END_MAZES.get(maz_path.stem.lower())
				start_cells = config.get("start_cells") if config else None
				end_cells = config.get("end_cells") if config else None
				maze.save_json(json_path, start_cells=start_cells, end_cells=end_cells)
				maze.save_jpeg(jpeg_path, start_cells=start_cells, end_cells=end_cells)
			except Exception as exc:  # pragma: no cover - batch export guard
				errors.append((maz_path, str(exc)))
		return errors


if __name__ == "__main__":
	binary_dir = PROJECT_ROOT / "mazefiles" / "binary"
	output_dir = PROJECT_ROOT / "mazefiles" / "json_with_images"
	print(f"Processing {binary_dir} -> {output_dir}")
	errors = MicromouseMaze.export_all(binary_dir, output_dir)
	if errors:
		print("\nCompleted with errors:")
		for path, message in errors:
			print(f" - {path}: {message}")
	else:
		print("\nAll mazes exported successfully.")
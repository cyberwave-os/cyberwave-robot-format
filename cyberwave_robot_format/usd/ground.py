# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Ground-visual cut-outs over buried, visual-only scenery.

The web viewer's floor is a grid that never occludes, so a visual-only body
sunk below ``z = 0`` (a photogrammetry scan whose ballast sits under the rail
tops) is fully visible on the website. Both simulators draw an opaque ground at
``z = 0`` instead, which reduces it to the strips that poke through.

This is the USD scene composer's (:mod:`.scene`) rule for restoring parity
with the website without touching physics: the ground's collider stays exactly
as it is; its *visual* is tiled around the XY footprint of every such body; and
a pit-floor patch with the ground's appearance closes each hole just below the
body, so ragged scan edges show floor rather than void. A scene with nothing
buried never reaches here.

Pure Python, no ``pxr``: the geometry is computed here and only
:mod:`.scene` writes it to the stage. The MuJoCo scene export does not apply it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

GROUND_VISUAL_HALF_EXTENT_M = 50.0
"""Half-size of the tiled ground visual: the USD ground plane's 100 m square."""

BURIED_EPSILON_M = 0.01
"""How far below ``z = 0`` a body must reach before the ground opens over it."""

PIT_FLOOR_CLEARANCE_M = 0.005
"""Gap between a buried body's lowest point and the pit floor patch under it."""

PIT_GROUND_MARGIN_M = 10.0
"""Ground kept around the farthest pit when the scene reaches past the square."""


@dataclass(frozen=True)
class GroundRect:
    """An axis-aligned, horizontal ground patch at height ``z`` (world frame, metres)."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    z: float = 0.0

    @property
    def center(self) -> tuple[float, float, float]:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0, self.z)

    @property
    def size(self) -> tuple[float, float]:
        return (self.max_x - self.min_x, self.max_y - self.min_y)


def buried_pit(lower: Sequence[float], upper: Sequence[float]) -> GroundRect | None:
    """The pit under a visual-only body whose world AABB is ``lower``..``upper``.

    ``None`` unless the body reaches more than :data:`BURIED_EPSILON_M` below the
    ground and has an XY footprint. The pit spans that footprint, its floor
    :data:`PIT_FLOOR_CLEARANCE_M` below the body's lowest point.
    """
    if not float(lower[2]) < -BURIED_EPSILON_M:
        return None
    if not (float(upper[0]) > float(lower[0]) and float(upper[1]) > float(lower[1])):
        return None
    return GroundRect(
        min_x=float(lower[0]),
        min_y=float(lower[1]),
        max_x=float(upper[0]),
        max_y=float(upper[1]),
        z=float(lower[2]) - PIT_FLOOR_CLEARANCE_M,
    )


def ground_visual_half_extent(pits: Sequence[GroundRect]) -> float:
    """Half-size of a ground square that covers every pit.

    At least :data:`GROUND_VISUAL_HALF_EXTENT_M`; a scene whose buried scenery
    reaches farther grows the square to keep :data:`PIT_GROUND_MARGIN_M` of
    ground beyond its farthest pit, so no pit is clipped and no floor vanishes
    next to it while the (unbounded) collider still extends there.
    """
    reach = max(
        (abs(v) for p in pits for v in (p.min_x, p.min_y, p.max_x, p.max_y)),
        default=0.0,
    )
    return max(GROUND_VISUAL_HALF_EXTENT_M, reach + PIT_GROUND_MARGIN_M)


def ground_visual_layout(
    pits: Sequence[GroundRect],
    half_extent: float | None = None,
) -> tuple[list[GroundRect], list[GroundRect]]:
    """Tile the ``±half_extent`` ground square around ``pits``.

    ``half_extent`` defaults to :func:`ground_visual_half_extent`. Returns ``(tiles, pit_floors)``: the ``z = 0`` tiles covering the square
    except every pit footprint, and the pits clipped to the square (a pit wholly
    outside it is dropped). The square is cut on every pit edge into a grid; open
    cells merge into runs along X, then identical runs merge along Y, so a lone
    pit yields four tiles. Deterministic for the same input.
    """
    if half_extent is None:
        half_extent = ground_visual_half_extent(pits)
    floors: list[GroundRect] = []
    for pit in pits:
        clipped = GroundRect(
            min_x=max(pit.min_x, -half_extent),
            min_y=max(pit.min_y, -half_extent),
            max_x=min(pit.max_x, half_extent),
            max_y=min(pit.max_y, half_extent),
            z=pit.z,
        )
        if clipped.max_x > clipped.min_x and clipped.max_y > clipped.min_y:
            floors.append(clipped)

    xs = sorted({-half_extent, half_extent, *(v for p in floors for v in (p.min_x, p.max_x))})
    ys = sorted({-half_extent, half_extent, *(v for p in floors for v in (p.min_y, p.max_y))})

    def is_open(x: float, y: float) -> bool:
        return not any(p.min_x < x < p.max_x and p.min_y < y < p.max_y for p in floors)

    tiles: list[GroundRect] = []
    # X-run -> the Y at which the tile currently growing over it started.
    growing: dict[tuple[float, float], float] = {}
    for j in range(len(ys) - 1):
        y = (ys[j] + ys[j + 1]) / 2.0
        runs: list[tuple[float, float]] = []
        for i in range(len(xs) - 1):
            if not is_open((xs[i] + xs[i + 1]) / 2.0, y):
                continue
            if runs and runs[-1][1] == xs[i]:
                runs[-1] = (runs[-1][0], xs[i + 1])
            else:
                runs.append((xs[i], xs[i + 1]))
        for run in [r for r in growing if r not in runs]:
            tiles.append(GroundRect(run[0], growing.pop(run), run[1], ys[j]))
        for run in runs:
            growing.setdefault(run, ys[j])
    for run, start in growing.items():
        tiles.append(GroundRect(run[0], start, run[1], ys[-1]))
    tiles.sort(key=lambda r: (r.min_y, r.min_x))
    return tiles, floors


__all__ = [
    "BURIED_EPSILON_M",
    "GROUND_VISUAL_HALF_EXTENT_M",
    "PIT_FLOOR_CLEARANCE_M",
    "PIT_GROUND_MARGIN_M",
    "GroundRect",
    "buried_pit",
    "ground_visual_half_extent",
    "ground_visual_layout",
]

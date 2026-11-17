# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Collision-shape helpers shared by the importers.

MuJoCo collides a mesh as its convex hull, so a hollow scene (a room, a tunnel, a
shelf) imported as one mesh collides as a solid block. :func:`hull_volume_ratio`
measures how far a mesh is from convex, and :func:`hollow_collision_warning` turns a
low ratio into the sentence shown to the uploader. :func:`fit_collision_boxes`
turns one mesh into the boxes that stand for it. None of them needs USD.
"""

from __future__ import annotations

import math

import numpy as np

#: Below this mesh-volume / hull-volume ratio a mesh is treated as hollow.
HOLLOW_HULL_RATIO = 0.35

#: The hull is built from at most this many vertices (a fixed-seed sample above it),
#: which bounds the cost on multi-hundred-thousand-triangle scenes.
_MAX_HULL_POINTS = 100_000


def hull_volume_ratio(vertices: np.ndarray, faces: np.ndarray) -> float | None:
    """Mesh volume divided by convex-hull volume, in ``[0, 1]``.

    The mesh volume is the signed sum over triangles (divergence theorem), so it does
    not need the mesh to be watertight; overlapping solids can only push it up, never
    down, so a hollow scene is never mistaken for a solid one. ``None`` when the
    ratio cannot be computed (degenerate input, or ``scipy`` is not installed).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) < 4 or len(faces) == 0:
        return None
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None

    tri = vertices[faces]
    volume = abs(
        float(np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
    )
    points = np.unique(np.round(vertices, 6), axis=0)
    if len(points) > _MAX_HULL_POINTS:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), _MAX_HULL_POINTS, replace=False)]
    try:
        hull_volume = float(ConvexHull(points).volume)
    except Exception:  # Qhull raises its own error type for flat/degenerate input
        return None
    if hull_volume <= 0:
        return None
    return min(1.0, volume / hull_volume)


def hollow_collision_warning(ratio: float | None) -> str | None:
    """The uploader-facing warning for a hollow mesh, or ``None`` if it is fine."""
    if ratio is None or ratio >= HOLLOW_HULL_RATIO:
        return None
    return (
        f"This asset looks hollow (its volume is {ratio:.0%} of its convex hull), "
        "and the collision will be a solid block."
    )


# ----------------------------------------------------------------------------
# Box fitting
# ----------------------------------------------------------------------------

#: Cell size of the grid a hollow mesh is cut into, in metres. Besides the gaps
#: below, which are closed on purpose, a gap up to two cells wider may close.
BOX_FIT_PITCH_M = 0.1

#: A vertical gap shorter than this between two parts of one mesh is closed: nothing
#: drives under a 0.8 m shelf, and a rack then is a few boxes instead of one per beam.
BOX_FIT_CLEARANCE_M = 1.0

#: A horizontal gap narrower than this is closed too: no robot fits through it, and
#: a wall is then one solid box instead of a box for each of its faces.
BOX_FIT_MIN_GAP_M = 0.4

#: A hollow mesh that needs more boxes than this is cut again on a coarser grid.
BOX_FIT_MAX_BOXES = 64

#: No box is thinner than this: a zero-size box is not a valid collider.
_MIN_BOX_SIZE_M = 0.01

#: Bounds on the work for one mesh. Past either, the grid is made coarser.
_MAX_GRID_CELLS = 4_000_000
_MAX_SURFACE_SAMPLES = 8_000_000

#: The grid is made coarser at most this many times (0.1 m -> 1.6 m).
_MAX_COARSENINGS = 4

#: A rectangle this close to the axis-aligned one in area is not worth a rotation.
_YAW_AREA_GAIN = 0.02

#: A closed part with at least this share of its box's volume is that box. Any
#: lower share may be a doorway or an aisle, which one box would close.
_FULL_BOX_RATIO = 0.999

#: Cells of an enclosed space tested for lying inside the part's material.
_CAVITY_PROBES = 5


def fit_collision_boxes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    pitch: float = BOX_FIT_PITCH_M,
    clearance: float = BOX_FIT_CLEARANCE_M,
    max_boxes: int = BOX_FIT_MAX_BOXES,
    notes: list[str] | None = None,
) -> list[dict[str, list[float]]]:
    """Boxes that stand for one mesh, as ``{"size", "xyz", "rpy"}`` in its frame.

    The frame is Z-up and in metres. A closed part that is a box (a crate, a pillar,
    a wall, a slab) is that box, turned about Z to fit a part that is not
    axis-aligned. Any other part (walls joined at a corner, a ring of walls, a
    building shell, a rack, a wall with a doorway) is cut into boxes that follow its
    surface, so the space it leaves open stays open; the inside of a closed part's
    material is filled. One bounding box for such a part would fill that space, and
    a robot could not enter it.

    Parts are the pieces of the mesh that share no vertex, as long as there are at
    most ``max_boxes`` of them; a mesh of more pieces is fitted as one part. If the
    parts need more than ``max_boxes`` boxes together, those that keep the least
    space open per box become one box each, until the mesh is within the limit.

    ``notes``, if given, gets a line for each place the boxes may close open space:
    a part that became one solid block, or one cut on a coarser grid.

    Returns ``[]`` for a mesh with no triangles or with non-finite coordinates.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        return []
    # Pieces that only touch are still joined: exporters repeat a vertex per face.
    welded = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)[1]
    welded = welded.reshape(-1)[faces]
    fits = [
        _fit_part(
            vertices[faces[part]], welded[part], pitch, clearance, max_boxes, notes
        )
        for part in _parts(welded, max_boxes)
    ]
    # The limit is the mesh's, not each piece's. A part made one box keeps the space
    # between the parts open, which a coarser grid over the whole mesh would close.
    excess = sum(len(boxes) for boxes, _ in fits) - max_boxes
    by_space_kept = sorted(
        (i for i, (boxes, _) in enumerate(fits) if len(boxes) > 1),
        key=lambda i: (
            (_volume([fits[i][1]]) - _volume(fits[i][0])) / (len(fits[i][0]) - 1)
        ),
    )
    collapsed = 0
    for i in by_space_kept:
        if excess <= 0:
            break
        boxes, whole = fits[i]
        excess -= len(boxes) - 1
        fits[i] = ([whole], whole)
        collapsed += 1
    if collapsed and notes is not None:
        notes.append(
            f"{collapsed} of its pieces collide as solid blocks, to keep it within "
            f"{max_boxes} collision boxes"
        )
    return [box for boxes, _ in fits for box in boxes]


def _volume(boxes: list[dict[str, list[float]]]) -> float:
    return sum(float(np.prod(box["size"])) for box in boxes)


def _parts(welded: np.ndarray, limit: int) -> list[np.ndarray]:
    """The faces of each piece of the mesh, or all of them if it cannot be split.

    ``welded`` is the faces, with the vertices that share a position merged.
    """
    every_face = [np.arange(len(welded))]
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return every_face
    count = int(welded.max()) + 1
    edges = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]]])
    graph = coo_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
        shape=(count, count),
    )
    label = connected_components(graph, directed=False)[1]
    # Counted over faces: a vertex no face uses is not a piece.
    pieces, face_label = np.unique(label[welded[:, 0]], return_inverse=True)
    if len(pieces) < 2 or len(pieces) > limit:
        return every_face
    face_label = face_label.reshape(-1)
    return [np.flatnonzero(face_label == piece) for piece in range(len(pieces))]


def _fit_part(
    tri: np.ndarray,
    welded: np.ndarray,
    pitch: float,
    clearance: float,
    max_boxes: int,
    notes: list[str] | None,
) -> tuple[list[dict[str, list[float]]], dict[str, list[float]]]:
    """The boxes for one part, given as its ``(n, 3, 3)`` triangles and welded faces.

    Also the one box around the part, for when the mesh has too many boxes.
    """
    yaw = _footprint_yaw(tri.reshape(-1, 3)[:, :2])
    cos, sin = math.cos(yaw), math.sin(yaw)
    to_world = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    tri = tri @ to_world  # row vectors: world -> the turned frame
    lo, hi = tri.min(axis=(0, 1)), tri.max(axis=(0, 1))

    box_volume = float(np.prod(hi - lo))
    # Only a closed part has a volume: walls with no roof, facing out, would come
    # out at two thirds of their box. Signed, so a room modelled as one
    # inward-facing shell does not count as full.
    closed = _is_closed(welded)
    volume = float(np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6
    if closed and box_volume > 0 and volume / box_volume >= _FULL_BOX_RATIO:
        bounds = [(lo, hi)]
    else:
        solid = tri if closed else None
        bounds = _surface_boxes(tri, lo, hi, pitch, clearance, max_boxes, solid, notes)
    whole = _box(lo, hi, to_world, yaw)
    return [_box(b_lo, b_hi, to_world, yaw) for b_lo, b_hi in bounds], whole


def _footprint_yaw(points: np.ndarray) -> float:
    """The turn about Z, in ``(-pi/4, pi/4]``, of the smallest rectangle around ``points``.

    ``0.0`` unless a turned rectangle is clearly smaller, so an axis-aligned scene
    keeps axis-aligned boxes.
    """
    points = np.unique(np.round(points, 6), axis=0)
    if len(points) < 2:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        # The hull is all the rectangle depends on, and far fewer points.
        candidates = points = points[ConvexHull(points).vertices]
    except ImportError:
        return 0.0
    except Exception:  # Qhull: the footprint is a line (a wall with no thickness)
        centred = points - points.mean(axis=0)
        direction = np.linalg.svd(centred, full_matrices=False)[2][0]
        candidates = np.array([points.mean(axis=0), points.mean(axis=0) + direction])

    edges = np.roll(candidates, -1, axis=0) - candidates
    edges = edges[np.linalg.norm(edges, axis=1) > 1e-9]
    if len(edges) == 0:
        return 0.0
    # A rectangle is the same one every quarter turn.
    angles = (np.arctan2(edges[:, 1], edges[:, 0]) + np.pi / 4) % (np.pi / 2)
    angles -= np.pi / 4

    def area(angle: float) -> float:
        cos, sin = math.cos(angle), math.sin(angle)
        u = points[:, 0] * cos + points[:, 1] * sin
        v = points[:, 1] * cos - points[:, 0] * sin
        return float((u.max() - u.min()) * (v.max() - v.min()))

    best = min(angles, key=area)
    # Compared on the longer side too: a wall with no thickness has no area at all.
    span = float(np.ptp(points, axis=0).max())
    if area(0.0) - area(best) <= _YAW_AREA_GAIN * max(area(0.0), span * 1e-3):
        return 0.0
    return float(best)


def _surface_boxes(
    tri: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    pitch: float,
    clearance: float,
    max_boxes: int,
    solid: np.ndarray | None,
    notes: list[str] | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut a mesh into boxes along its surface, on a grid of ``pitch``.

    ``solid`` is the part's triangles if it is closed: the space it encloses is then
    filled where it lies inside its material.
    """
    edge = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2).max(axis=1)
    for _ in range(_MAX_COARSENINGS + 1):
        # The last cell holds the far face: a gap closed against the grid's end then
        # stops at the part.
        shape = np.maximum(np.ceil((hi - lo) / pitch - 1e-6), 1).astype(np.int64)
        # A point every half cell along each triangle, so no cell it crosses is missed.
        steps = np.maximum(1, np.ceil(edge / (pitch / 2))).astype(np.int64)
        steps = 2 ** np.ceil(np.log2(steps)).astype(np.int64)
        samples = int(((steps + 1) * (steps + 2) // 2).sum())
        if np.prod(shape) <= _MAX_GRID_CELLS and samples <= _MAX_SURFACE_SAMPLES:
            points = _sample_triangles(tri, steps)
            # A face that lies on a cell boundary goes to the cell below it, all of
            # it: left to rounding it would land on both sides and tear the box up.
            cells = np.floor((points - lo) / pitch - 1e-6).astype(np.int64)
            cells = np.clip(cells, 0, shape - 1)
            bounds = _grid_boxes(
                points,
                cells,
                shape,
                lo,
                pitch,
                round(clearance / pitch),
                max_boxes,
                solid,
            )
            if len(bounds) <= max_boxes:
                # A gap up to two cells wider than the one closed on purpose may
                # close; from 0.4 m cells that is wider than a robot.
                gap_cells = round(BOX_FIT_MIN_GAP_M / pitch)
                if gap_cells <= 1 and notes is not None:
                    notes.append(
                        f"its collision boxes were cut on a {pitch:g} m grid, so "
                        f"gaps narrower than about {(gap_cells + 2) * pitch:g} m "
                        "may be closed"
                    )
                return [(b_lo, np.minimum(b_hi, hi)) for b_lo, b_hi in bounds]
        pitch *= 2
    # Too large or too intricate for boxes (a terrain, say): its bounding box, which
    # is what every mesh got before. Boxes past the limit would crowd out the rest
    # of the scene's colliders.
    if notes is not None:
        notes.append(
            "it is too intricate for collision boxes and collides as a solid block"
        )
    return [(lo, hi)]


def _sample_triangles(tri: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Points on each triangle, on a barycentric grid of ``steps`` per edge."""
    out = []
    for n in np.unique(steps):
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        a, b = i[keep] / n, j[keep] / n
        weights = np.stack([a, b, 1.0 - a - b], axis=1)
        out.append(np.einsum("kc,tcd->tkd", weights, tri[steps == n]).reshape(-1, 3))
    return np.concatenate(out)


def _grid_boxes(
    points: np.ndarray,
    cells: np.ndarray,
    shape: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    clearance_cells: int,
    max_boxes: int,
    solid: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Merge the cells the surface touches into boxes, each shrunk onto the surface.

    Stops past ``max_boxes``, a grid the caller rejects.
    """
    nx, ny, nz = (int(n) for n in shape)
    flat = (cells[:, 0] * ny + cells[:, 1]) * nz + cells[:, 2]
    order = np.argsort(flat, kind="stable")
    flat, points = flat[order], points[order]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    touched = flat[starts]
    cell_lo = np.minimum.reduceat(points, starts, axis=0)
    cell_hi = np.maximum.reduceat(points, starts, axis=0)
    cell_index = np.stack(np.unravel_index(touched, (nx, ny, nz)), axis=1)

    occupied = np.zeros((nx, ny, nz), dtype=bool)
    occupied.reshape(-1)[touched] = True
    # Nothing passes between two touched cells that are closer than a robot is tall
    # (in a column) or wide (in a row), so those gaps are closed. Nor between one
    # and the box's bottom or sides: a round tank stays one box.
    _close_gaps(occupied, 2, clearance_cells, (True, False))
    for axis in (0, 1):
        _close_gaps(occupied, axis, round(BOX_FIT_MIN_GAP_M / pitch), (True, True))
    if solid is not None:
        _fill_inside(occupied, solid, origin, pitch)

    bounds = []
    remaining = occupied.reshape(-1)
    start = 0
    while True:
        offset = int(remaining[start:].argmax())
        if not remaining[start + offset]:
            break
        start += offset
        x, y, z = (int(v) for v in np.unravel_index(start, (nx, ny, nz)))
        x1, y1, z1 = x + 1, y + 1, z + 1
        while x1 < nx and occupied[x1, y, z]:
            x1 += 1
        while y1 < ny and occupied[x:x1, y1, z].all():
            y1 += 1
        while z1 < nz and occupied[x:x1, y:y1, z1].all():
            z1 += 1
        occupied[x:x1, y:y1, z:z1] = False
        first, last = np.array([x, y, z]), np.array([x1, y1, z1]) - 1
        inside = ((cell_index >= first) & (cell_index <= last)).all(axis=1)
        index, low, high = cell_index[inside], cell_lo[inside], cell_hi[inside]
        box_lo, box_hi = origin + first * pitch, origin + (last + 1) * pitch
        # Each face moves in to the surface inside its outermost layer of cells, so
        # a thin wall keeps its thickness. A layer with only closed gaps stays put.
        for axis in range(3):
            near, far = index[:, axis] == first[axis], index[:, axis] == last[axis]
            if near.any():
                box_lo[axis] = low[near, axis].min()
            if far.any():
                box_hi[axis] = high[far, axis].max()
        bounds.append((box_lo, box_hi))
        if len(bounds) > max_boxes:
            break
    return bounds


def _is_closed(welded: np.ndarray) -> bool:
    """Does every edge of the part run as often one way as the other?

    True for closed solids, also ones that share an edge with another; false for
    a sheet, or for walls with no roof.
    """
    start, end = welded.reshape(-1), welded[:, [1, 2, 0]].reshape(-1)
    keep = start != end
    start, end = start[keep], end[keep]  # copies, so free to change in place
    count = int(welded.max()) + 1
    forward = start * count + end
    forward.sort()
    end *= count
    end += start  # each edge the other way round
    end.sort()
    return len(forward) > 0 and np.array_equal(forward, end)


def _fill_inside(
    occupied: np.ndarray, tri: np.ndarray, origin: np.ndarray, pitch: float
) -> None:
    """Fill, in place, each enclosed space that lies inside the part's material.

    A solid given by its faces (a crate, a rack row) encloses its own inside; a
    building with a roof encloses a room. Only the first is filled.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return
    spaces, count = ndimage.label(~occupied)
    reaches_out = np.zeros(count + 1, dtype=bool)
    for axis in range(3):
        for end in (0, -1):
            reaches_out[np.take(spaces, end, axis=axis)] = True
    probes, owner = [], []
    for space, where in enumerate(ndimage.find_objects(spaces), start=1):
        if where is None or reaches_out[space]:
            continue
        cells = np.argwhere(spaces[where] == space)
        n = min(_CAVITY_PROBES, len(cells))
        pick = cells[np.linspace(0, len(cells) - 1, n).astype(np.int64)]
        probes.append(pick + [s.start for s in where])
        owner += [space] * n
    if not probes:
        return
    # Off the cell centre, so the ray does not run along an edge on the grid.
    points = origin + (np.concatenate(probes) + [0.5123, 0.5234, 0.5]) * pitch
    inside = _inside(tri, points)
    owner_array = np.asarray(owner)
    # Every probe of a space must agree: a room filled by mistake is a solid block.
    filled = np.setdiff1d(owner_array, owner_array[~inside])
    occupied |= np.isin(spaces, filled)


def _inside(tri: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Is each point inside the closed part? Counted along a ray straight up."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    # Twice the face's area seen from above: positive where the face points up.
    up = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    keep = up != 0  # a vertical face runs along the ray, never across it
    a, b, c, up = a[keep], b[keep], c[keep], up[keep]
    xy_lo = np.minimum(np.minimum(a, b), c)[:, :2]
    xy_hi = np.maximum(np.maximum(a, b), c)[:, :2]
    out = np.zeros(len(points), dtype=bool)
    for i, (x, y, z) in enumerate(points):
        near = np.flatnonzero(
            (xy_lo[:, 0] <= x)
            & (xy_hi[:, 0] >= x)
            & (xy_lo[:, 1] <= y)
            & (xy_hi[:, 1] >= y)
        )
        pa, pb, pc, d = a[near], b[near], c[near], up[near]
        wa = ((pb[:, 0] - x) * (pc[:, 1] - y) - (pb[:, 1] - y) * (pc[:, 0] - x)) / d
        wb = ((pc[:, 0] - x) * (pa[:, 1] - y) - (pc[:, 1] - y) * (pa[:, 0] - x)) / d
        wc = 1.0 - wa - wb
        above = (wa >= 0) & (wb >= 0) & (wc >= 0)
        above &= wa * pa[:, 2] + wb * pb[:, 2] + wc * pc[:, 2] > z
        # The ray leaves the part through a face that points up.
        out[i] = np.sign(d[above]).sum() > 0
    return out


def _close_gaps(
    occupied: np.ndarray, axis: int, max_gap: int, ends: tuple[bool, bool]
) -> None:
    """Fill, in place, runs of at most ``max_gap`` free cells between two occupied.

    ``ends`` says whether the grid's first and last cell along ``axis`` count as
    occupied too. A run that reaches both ends stays free: it passes through.
    """
    if max_gap < 1:
        return
    cells = np.moveaxis(occupied, axis, -1)  # a view: writes land in ``occupied``
    n = cells.shape[-1]
    index = np.arange(n)
    before = np.maximum.accumulate(np.where(cells, index, -1), axis=-1)
    after = np.minimum.accumulate(np.where(cells, index, n)[..., ::-1], axis=-1)
    after = after[..., ::-1]
    low, high = before >= 0, after < n
    cells |= (
        (low | ends[0])
        & (high | ends[1])
        & (low | high)
        & (after - before - 1 <= max_gap)
    )


def _box(
    lo: np.ndarray, hi: np.ndarray, to_world: np.ndarray, yaw: float
) -> dict[str, list[float]]:
    return {
        "size": np.maximum(hi - lo, _MIN_BOX_SIZE_M).tolist(),
        "xyz": (to_world @ ((lo + hi) / 2)).tolist(),
        "rpy": [0.0, 0.0, yaw],
    }

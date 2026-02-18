"""
Mesh URI resolution utilities.

Handles resolution of mesh file paths including ROS-style package:// URIs.

# TODO: urdf_projects worked before, but we should find a proper place and name for it
In Cyberwave, URDF files are stored in /app/media/urdf_projects/{slug}/ and
mesh files are typically in subdirectories like meshes/. The backend rewrites
package:// URIs to relative paths during upload, so most resolution is
simply joining the base_dir with the relative path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import resolve_robotics_uri_py as rru


logger = logging.getLogger(__name__)


def _package_search_paths(base_dir: Path) -> list[Path]:
    """Build list of directories to search for ROS packages.

    Returns a list of potential package directories based on the base directory.
    Searches the URDF directory and its parents.
    """
    paths: list[Path] = [base_dir]
    paths.extend(base_dir.parents)
    return paths


def resolve_mesh_uri(filename: str, base_dir: Path) -> str:
    """Resolve mesh filenames including ROS-style package:// URIs.

    Resolution strategy:
    1. For package:// URIs: use resolve-robotics-uri-py, then fallback to
       searching base_dir and its parents
    2. For relative paths: resolve relative to base_dir
    3. For absolute paths: verify existence, fallback to relative search

    Args:
        filename: The mesh filename or URI to resolve
        base_dir: The directory containing the URDF file

    Returns:
        Resolved absolute path to the mesh file, or original filename if not found
    """
    if filename.startswith("package://"):
        package_dirs = [str(p) for p in _package_search_paths(base_dir)]
        try:
            resolved = rru.resolve_robotics_uri(filename, package_dirs=package_dirs)
            if resolved:
                return str(resolved)
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "Failed to resolve %s via resolve-robotics-uri-py: %s", filename, exc
            )

    if filename.startswith("package://"):
        package_path = filename[len("package://") :]
        parts = package_path.split("/", 1)
        package_name = parts[0]
        relative_path = parts[1] if len(parts) > 1 else ""
        for root in _package_search_paths(base_dir):
            candidates = [
                root / package_name / relative_path
                if package_name
                else root / relative_path,
                root / relative_path,
            ]
            for cand in candidates:
                if cand.exists():
                    return str(cand)

        # Fallback: Recursive search for the filename in base_dir and parent
        # This handles malformed package URIs or missing intermediate directories
        target_name = Path(relative_path).name
        search_dirs = [base_dir, base_dir.parent]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            try:
                matches = list(search_dir.rglob(target_name))
                if matches:
                    logger.info(
                        "Resolved %s via recursive search to %s", filename, matches[0]
                    )
                    return str(matches[0])
            except Exception:
                pass

        logger.warning("Mesh file not found: %s", filename)
        return filename

    path_obj = Path(filename)
    if not path_obj.is_absolute():
        full_path = base_dir / path_obj
        if full_path.exists():
            return str(full_path)
        logger.warning("Mesh file not found: %s", filename)

        # Fallback: recursive search in base_dir and parent
        # This handles cases like "visual/foo.stl" where folder structure is
        # "meshes/visual/foo.stl" or cases where the path is relative to a
        # parent directory
        search_dirs = [base_dir, base_dir.parent]
        target_name = path_obj.name

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            try:
                # Find file with same name
                matches = list(search_dir.rglob(target_name))
                if matches:
                    logger.info(
                        "Resolved %s via recursive search to %s", filename, matches[0]
                    )
                    return str(matches[0])
            except Exception:
                pass

        return filename

    # Handle fake absolute paths (e.g. starting with / but relative, or /../..)
    # If absolute path doesn't exist, try treating it as relative
    if path_obj.is_absolute() and not path_obj.exists():
        # Strip leading slash/drive
        rel_candidate = str(filename).lstrip(os.sep)
        full_path = base_dir / rel_candidate
        if full_path.exists():
            logger.info(
                "Resolved absolute-looking path %s as relative to %s",
                filename,
                full_path,
            )
            return str(full_path)

        # Recursive search for the filename as last resort
        search_dirs = [base_dir, base_dir.parent]
        target_name = path_obj.name
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            try:
                matches = list(search_dir.rglob(target_name))
                if matches:
                    logger.info(
                        "Resolved %s via recursive search to %s", filename, matches[0]
                    )
                    return str(matches[0])
            except Exception:
                pass

    return filename


__all__ = ["resolve_mesh_uri"]

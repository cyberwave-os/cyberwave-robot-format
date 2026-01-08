# Copyright [2025] Tomáš Macháček <tomasmachacekw@gmail.com>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MJCF-specific utility functions."""

from __future__ import annotations

import logging

import numpy as np

from cyberwave_robot_format.schema import Link

logger = logging.getLogger(__name__)


def sanitize_inertia(link: Link) -> tuple[float, float, float, float, float, float]:
    """Ensure inertia tensor is positive definite for MuJoCo.

    MuJoCo requires valid inertia tensors. This function checks and fixes
    common issues like non-positive diagonal elements or ill-conditioning.

    Args:
        link: Link object with inertia and mass properties.

    Returns:
        Tuple of (ixx, iyy, izz, ixy, ixz, iyz) with sanitized values.
    """
    if not link.inertia:
        val = max(1e-6, link.mass * 1e-4)
        return val, val, val, 0.0, 0.0, 0.0

    ixx, iyy, izz = link.inertia.ixx, link.inertia.iyy, link.inertia.izz
    ixy, ixz, iyz = link.inertia.ixy, link.inertia.ixz, link.inertia.iyz
    mass = link.mass

    min_inertia = max(1e-6, mass * 1e-4)

    ixx = max(ixx, min_inertia)
    iyy = max(iyy, min_inertia)
    izz = max(izz, min_inertia)

    matrix = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    eigvals = np.linalg.eigvals(matrix)

    max_eig = np.max(np.abs(eigvals))
    min_eig = np.min(eigvals)

    is_ill_conditioned = max_eig > 1e6 * max(min_eig, 1e-12)

    if min_eig <= 0 or is_ill_conditioned:
        safe_val = max(min_inertia, mass * 1e-3)
        ixx, iyy, izz = safe_val, safe_val, safe_val
        ixy, ixz, iyz = 0.0, 0.0, 0.0

        reason = "non-positive definite" if min_eig <= 0 else "ill-conditioned"
        logger.warning(
            "Fixed invalid inertia for link '%s': %s (forced to diagonal %.2e).", link.name, reason, safe_val
        )

    diags = sorted([ixx, iyy, izz])
    if diags[0] + diags[1] < diags[2]:
        required_sum = diags[2] * 1.01
        diff = required_sum - (diags[0] + diags[1])
        boost = diff / 2.0

        if ixx == diags[0]:
            ixx += boost
        elif iyy == diags[0]:
            iyy += boost
        else:
            izz += boost

        if ixx == diags[1]:
            ixx += boost
        elif iyy == diags[1]:
            iyy += boost
        else:
            izz += boost

        logger.warning("Fixed inertia triangle inequality for link '%s'", link.name)

    return ixx, iyy, izz, ixy, ixz, iyz


__all__ = ["sanitize_inertia"]

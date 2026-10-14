# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""USD (OpenUSD / Universal Scene Description) parser and exporter.

``CommonSchema`` <-> USD, covering the full schema: links with inertials, the
kinematic tree as ``UsdPhysics`` joints, visual and collision geometry, materials,
actuators, sensors, contacts, collision filtering, and world physics. Unlike
:mod:`cyberwave_robot_format.mesh`, which deals in static geometry only, this
module preserves the articulation.

Requires the optional ``pxr`` module (``pip install cyberwave-robot-format[usd]``,
or ``pip install usd-core``). The import is deferred, so this module imports fine
without it — only calling ``export``/``parse`` raises.

Three principles govern the mapping, and understanding them explains every
decision in the exporter and parser:

**1. Dual encoding with a declared authority.** Where ``UsdPhysics`` has a
faithful, unambiguous home for a schema field, that is the only place it is
written. Where the native encoding is *lossy* (nearly all of ``UsdPhysics`` is
``float32``), *ambiguous* (a ``PhysicsRevoluteJoint`` with no limits could be
``REVOLUTE`` or ``CONTINUOUS``) or *absent* (mimic joints, armature, jerkmax,
motor electricals), the exporter writes both the native attribute — so Isaac Sim,
Omniverse and usdview see a correct, simulatable stage — and a
``cyberwave:``-namespaced double-precision sidecar that the parser treats as
authoritative. The sidecars are what make the round trip exact; the native
attributes are what make the output useful to anyone else.

**2. ``None`` means unauthored.** An optional schema field is authored only when
it is not ``None``, and read only when ``HasAuthoredValue()``. This is what keeps
``mu_static=None`` from returning as ``0.0``, and keeps ``UsdPhysics`` sentinels
(``physics:mass = 0`` = "derive from density", ``physics:lowerLimit = -inf`` =
"unlimited") out of the schema. The one distinction USD cannot express this way —
``limits=None`` versus a ``JointLimits()`` whose every field is ``None`` — gets an
explicit ``cyberwave:joint:hasLimits`` presence marker.

**3. Names are sanitized forward only.** USD prim names must be valid
identifiers; schema names are not. Rather than an escape grammar (which is not
injective without also escaping ``_``, and destroys the readability that is the
whole point of ``.usda``), names are sanitized and the original is carried in a
``cyberwave:name`` sidecar whenever sanitization changed anything.

Two structural consequences worth knowing before reading the code:

* ``UsdPhysics`` **forbids nested rigid bodies**, so link prims are flat siblings
  under ``/<Robot>/Links``, each carrying its accumulated world transform at the
  zero configuration — not a nested tree the way MJCF bodies are.
* ``UsdPhysics`` restricts ``physics:axis`` to ``"X"``/``"Y"``/``"Z"``, so an
  arbitrary schema axis is baked into the joint frame rotations
  ``physics:localRot0``/``localRot1`` and the token is always ``"X"``.

:func:`export_scene_usd` places several robot instances in one scene bundle (a
root layer referencing one shared asset layer per distinct robot) and returns
the identity map from instances, links, joints and sensors to prim paths -- the
USD counterpart of :meth:`CommonSchema.merge_in` plus the MJCF scene export.
What an instance stands for (a twin, a fixture, a test robot) is the caller's
business, exactly as it is for ``merge_in``.
"""

from cyberwave_robot_format.usd.exporter import USDExporter
from cyberwave_robot_format.usd.parser import USDParser
from cyberwave_robot_format.usd.scene import (
    SceneExportOptions,
    SceneInstance,
    StaticBoxFixture,
    UsdExportResult,
    ViewportOptions,
    export_scene_usd,
)
from cyberwave_robot_format.usd.scene_to_glb import (
    SceneGlbReport,
    SceneGlbResult,
    USDSceneToGLB,
    convert_usd_scene_to_glb,
)

__all__ = [
    "SceneExportOptions",
    "SceneGlbReport",
    "SceneGlbResult",
    "SceneInstance",
    "StaticBoxFixture",
    "USDExporter",
    "USDParser",
    "USDSceneToGLB",
    "UsdExportResult",
    "ViewportOptions",
    "convert_usd_scene_to_glb",
    "export_scene_usd",
]

"""``max_velocity`` on a POSITION actuator must not reach the MJCF.

The field is consumed by the exporter for *velocity* actuators only, where it
becomes the ``ctrlrange`` speed band. On a position actuator it is carried as
metadata for downstream consumers — ``cyberwave-sim``'s ramped-target speed
governor reads it off the twin's universal schema — and the exported scene must
be identical with and without it.

That equivalence is what makes the governor safe to opt into per asset: setting
the field cannot change physics, only the setpoint the simulator feeds in. If
it ever leaked into ``ctrlrange`` here, a bogie declaring 2 m/s would have its
+/-100 m of travel silently clamped to +/-2 m.
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from cyberwave_robot_format.mjcf import MJCFExporter
from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    CommonSchema,
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
    Metadata,
    Vector3,
    Visual,
)


def _rail_schema(max_velocity):
    """The rail bogie's contract: a prismatic joint on a position actuator."""
    return CommonSchema(
        metadata=Metadata(name="rail_bogie"),
        links=[
            Link(name="rail_base_link", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(
                name="bogie_frame_link",
                mass=1500.0,
                inertia=Inertia(ixx=52.9, iyy=57.8, izz=96.9),
                visuals=[
                    Visual(
                        name="frame",
                        geometry=Geometry(
                            type=GeometryType.BOX, size=Vector3(1.0, 0.5, 0.3)
                        ),
                    )
                ],
            ),
        ],
        joints=[
            Joint(
                name="rail_joint",
                type=JointType.PRISMATIC,
                parent_link="rail_base_link",
                child_link="bogie_frame_link",
                axis=Vector3(1, 0, 0),
                limits=JointLimits(lower=-100.0, upper=100.0, effort=20000.0, velocity=2.0),
            )
        ],
        actuators=[
            Actuator(
                name="rail_joint_position",
                joint="rail_joint",
                type=ActuatorType.POSITION,
                max_torque=20000.0,
                kp=100000.0,
                kd=6000.0,
                max_velocity=max_velocity,
            )
        ],
    )


def _export(schema):
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "scene.xml"
        exporter.export(schema, output)
        return output.read_text()


def test_position_actuator_max_velocity_does_not_change_the_mjcf():
    """Declaring the governor is byte-identical to not declaring it."""
    assert _export(_rail_schema(2.0)) == _export(_rail_schema(None))


def test_position_actuator_ctrlrange_still_comes_from_joint_travel():
    """The speed cap must not be mistaken for a control range.

    A position actuator's ctrl IS a position, so its ctrlrange bounds travel.
    Writing the speed band here would clamp the bogie to 2 m of rail.
    """
    root = ET.fromstring(_export(_rail_schema(2.0)))
    actuator = root.find(".//actuator/position[@name='rail_joint_position']")

    assert actuator is not None
    ctrl_lo, ctrl_hi = (float(v) for v in actuator.get("ctrlrange").split())
    assert (ctrl_lo, ctrl_hi) == (-100.0, 100.0)
    assert actuator.get("forcerange") == "-20000.0 20000.0"

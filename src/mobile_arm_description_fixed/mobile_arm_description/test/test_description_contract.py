#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import subprocess
import unittest
import xml.etree.ElementTree as ET

import PyKDL
import yaml
from kdl_parser_py.urdf import treeFromString


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XACRO_PATH = os.path.join(PACKAGE_DIR, "urdf", "mobile_arm.urdf.xacro")
GAZEBO_CONTROL_PATH = os.path.join(
    PACKAGE_DIR,
    "launch",
    "gazebo_control.launch",
)
GAZEBO_PLANAR_CONTROL_PATH = os.path.join(
    PACKAGE_DIR,
    "launch",
    "gazebo_planar_control.launch",
)
PLANAR_CONTROLLER_PATH = os.path.join(
    PACKAGE_DIR,
    "config",
    "mobile_arm_planar_controllers.yaml",
)
VIRTUAL_JOINTS = ["x", "y", "z", "sway"]
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
SAFE_INITIAL_ARM_POSE = {
    "initial_joint1": "0.0",
    "initial_joint2": "-0.30",
    "initial_joint3": "0.03",
    "initial_joint4": "0.0",
    "initial_joint5": "0.03",
    "initial_joint6": "0.0",
}


def render(
        fixed_base,
        chassis_gravity=False,
        simple_collision=False,
        base_sensors=False,
        frictionless_planar_contacts=False,
        disable_arm_collisions=False):
    return subprocess.check_output([
        "xacro", XACRO_PATH,
        "fixed_base_diagnostic:={}".format(str(fixed_base).lower()),
        "disable_arm_gravity:=true",
        "disable_chassis_gravity:={}".format(str(chassis_gravity).lower()),
        "simple_chassis_collision:={}".format(str(simple_collision).lower()),
        "enable_base_sensors:={}".format(str(base_sensors).lower()),
        "frictionless_planar_contacts:={}".format(
            str(frictionless_planar_contacts).lower()
        ),
        "disable_arm_collisions:={}".format(
            str(disable_arm_collisions).lower()
        ),
    ])


class DescriptionContractTest(unittest.TestCase):
    def test_gazebo_launch_enables_idle_position_hold(self):
        root = ET.parse(GAZEBO_CONTROL_PATH).getroot()
        arguments = dict(
            (argument.get("name"), argument.get("default"))
            for argument in root.findall("arg")
        )
        self.assertEqual(arguments.get("enable_arm_idle_hold"), "true")
        node = root.find(
            "./node[@type='arm_idle_position_hold.py']"
        )
        self.assertIsNotNone(node)
        self.assertEqual(node.get("if"), "$(arg enable_arm_idle_hold)")
        parameters = dict(
            (parameter.get("name"), parameter.get("value"))
            for parameter in node.findall("param")
        )
        self.assertEqual(parameters["command_timeout"], "0.30")
        self.assertEqual(parameters["control_rate"], "50.0")

    def test_launches_share_clearance_verified_initial_arm_pose(self):
        for launch_path in (
                GAZEBO_CONTROL_PATH,
                GAZEBO_PLANAR_CONTROL_PATH):
            root = ET.parse(launch_path).getroot()
            arguments = dict(
                (argument.get("name"), argument.get("default"))
                for argument in root.findall("arg")
            )
            for name, expected in SAFE_INITIAL_ARM_POSE.items():
                self.assertEqual(arguments.get(name), expected)

    def test_fixed_mode_removes_virtual_degrees_and_transmissions(self):
        xml = render(True)
        root = ET.fromstring(xml)
        joint_types = dict(
            (joint.get("name"), joint.get("type"))
            for joint in root.findall("joint")
        )
        for name in VIRTUAL_JOINTS:
            self.assertEqual(joint_types[name], "fixed")

        transmissions = set(
            transmission.get("name")
            for transmission in root.findall("transmission")
        )
        for name in VIRTUAL_JOINTS:
            self.assertNotIn(name + "_trans", transmissions)
        for name in ARM_JOINTS:
            self.assertIn(name + "_trans", transmissions)

    def test_planar_mode_exposes_internal_xy_and_yaw_pose(self):
        xml = render(False)
        root = ET.fromstring(xml)
        joint_types = dict(
            (joint.get("name"), joint.get("type"))
            for joint in root.findall("joint")
        )
        self.assertEqual(joint_types["x"], "prismatic")
        self.assertEqual(joint_types["y"], "prismatic")
        self.assertEqual(joint_types["z"], "continuous")
        self.assertEqual(joint_types["sway"], "fixed")

        transmissions = set(
            transmission.get("name")
            for transmission in root.findall("transmission")
        )
        for name in ("x", "y", "z"):
            self.assertIn(name + "_trans", transmissions)
        self.assertNotIn("sway_trans", transmissions)

        joints = dict(
            (joint.get("name"), joint)
            for joint in root.findall("joint")
        )
        self.assertEqual(joints["x"].find("limit").get("lower"), "-2.0")
        self.assertEqual(joints["x"].find("limit").get("upper"), "2.0")
        self.assertEqual(joints["y"].find("limit").get("lower"), "-2.0")
        self.assertEqual(joints["y"].find("limit").get("upper"), "2.0")

        # x/y are internal world-pose coordinates. Policies cannot command
        # them independently; the adapter maps [v, omega] through current yaw.
        self.assertEqual(joints["x"].find("origin").get("rpy"), "0 0 0")
        self.assertEqual(joints["x"].find("axis").get("xyz"), "1 0 0")
        self.assertEqual(joints["y"].find("origin").get("rpy"), "0 0 0")
        self.assertEqual(joints["y"].find("axis").get("xyz"), "0 1 0")
        self.assertEqual(joints["z"].find("origin").get("rpy"), "0 0 0")
        self.assertEqual(joints["z"].find("axis").get("xyz"), "0 0 1")
        self.assertIsNone(joints["z"].find("limit").get("lower"))
        self.assertIsNone(joints["z"].find("limit").get("upper"))
        self.assertEqual(joints["x"].find("limit").get("effort"), "20.0")
        self.assertEqual(joints["y"].find("limit").get("effort"), "20.0")
        self.assertEqual(joints["z"].find("limit").get("effort"), "10.0")
        self.assertEqual(
            joints["sway"].find("origin").get("xyz"),
            "0 0.3 0",
        )
        self.assertEqual(
            joints["sway"].find("origin").get("rpy"),
            "-1.5708 0 -1.5708",
        )
        self.assertEqual(
            joints["base_footprint_joint"].find("parent").get("link"),
            "z-link",
        )
        self.assertEqual(
            joints["base_footprint_joint"].find("child").get("link"),
            "base_footprint",
        )
        self.assertEqual(
            joints["base_footprint_joint"].find("origin").get("rpy"),
            "0 0 -1.5708",
        )

    def test_planar_velocity_gains_match_bounded_effort_contract(self):
        with open(PLANAR_CONTROLLER_PATH, "r") as stream:
            configuration = yaml.safe_load(stream)
        gains = configuration["gazebo_ros_control"]["pid_gains"]
        self.assertEqual(gains["x"], {"p": 100.0, "i": 0.0, "d": 0.0})
        self.assertEqual(gains["y"], {"p": 100.0, "i": 0.0, "d": 0.0})
        self.assertEqual(gains["z"], {"p": 20.0, "i": 0.0, "d": 0.0})

    def test_planar_spawn_starts_with_physical_forward_on_world_x(self):
        root = ET.parse(GAZEBO_CONTROL_PATH).getroot()
        arguments = dict(
            (argument.get("name"), argument.get("default"))
            for argument in root.findall("arg")
        )
        self.assertEqual(arguments["initial_planar_yaw"], "1.5708")
        fixed_spawn = root.find(
            "./node[@name='spawn_mobile_arm_fixed']"
        )
        planar_spawn = root.find(
            "./node[@name='spawn_mobile_arm_planar']"
        )
        self.assertIsNotNone(fixed_spawn)
        self.assertIsNotNone(planar_spawn)
        self.assertEqual(fixed_spawn.get("if"), "$(arg fixed_base_diagnostic)")
        self.assertEqual(
            planar_spawn.get("unless"),
            "$(arg fixed_base_diagnostic)",
        )
        self.assertIn("-J z $(arg initial_planar_yaw)", planar_spawn.get("args"))

    def test_fixed_kdl_chain_and_position_jacobian_are_six_dof(self):
        xml = render(True)
        ok, tree = treeFromString(xml)
        self.assertTrue(ok)
        chain = tree.getChain("base_link", "link6")
        chain_joints = []
        for index in range(chain.getNrOfSegments()):
            joint = chain.getSegment(index).getJoint()
            if joint.getTypeName() != "None":
                chain_joints.append(joint.getName())
        self.assertEqual(chain_joints, ARM_JOINTS)
        self.assertEqual(chain.getNrOfJoints(), 6)

        positions = PyKDL.JntArray(6)
        jacobian = PyKDL.Jacobian(6)
        solver = PyKDL.ChainJntToJacSolver(chain)
        self.assertEqual(solver.JntToJac(positions, jacobian), 0)
        position_jacobian = [
            [jacobian[row, column] for column in range(6)]
            for row in range(3)
        ]
        self.assertEqual(len(position_jacobian), 3)
        self.assertTrue(all(len(row) == 6 for row in position_jacobian))

    def test_gravity_and_collision_switches_change_only_requested_tags(self):
        full_mesh = ET.fromstring(render(False, False, False))
        no_chassis_gravity = ET.fromstring(render(False, True, False))
        simple_collision = ET.fromstring(render(False, False, True))

        def chassis_gravity_tags(root):
            return [
                gazebo.find("turnGravityOff").text
                for gazebo in root.findall("gazebo")
                if gazebo.get("reference") == "sway_link"
                and gazebo.find("turnGravityOff") is not None
            ]

        self.assertEqual(chassis_gravity_tags(full_mesh), [])
        self.assertEqual(chassis_gravity_tags(no_chassis_gravity), ["true"])

        full_collision = full_mesh.find("./link[@name='sway_link']/collision/geometry")
        simple_geometry = simple_collision.find(
            "./link[@name='sway_link']/collision/geometry"
        )
        self.assertIsNotNone(full_collision.find("mesh"))
        simple_box = simple_geometry.find("box")
        self.assertIsNotNone(simple_box)
        self.assertEqual(simple_box.get("size"), "1.12 0.20 0.29")
        simple_origin = simple_collision.find(
            "./link[@name='sway_link']/collision/origin"
        )
        self.assertEqual(simple_origin.get("xyz"), "0.43 0 0")

    def test_planar_sensors_publish_scan_and_contact_topics(self):
        root = ET.fromstring(render(False, True, True, True))
        laser_link = root.find("./link[@name='base_laser']")
        laser_joint = root.find("./joint[@name='base_laser_joint']")
        self.assertIsNotNone(laser_link)
        self.assertIsNotNone(laser_joint)
        self.assertEqual(
            laser_joint.find("parent").get("link"),
            "base_footprint",
        )
        self.assertEqual(
            laser_joint.find("origin").get("xyz"),
            "0.13 0 1.15",
        )
        self.assertEqual(
            laser_joint.find("origin").get("rpy"),
            "0 0 0",
        )

        laser_gazebo = root.find("./gazebo[@reference='base_laser']")
        laser_sensor = laser_gazebo.find("./sensor[@type='ray']")
        self.assertIsNotNone(laser_sensor)
        self.assertEqual(
            laser_sensor.find("plugin/topicName").text,
            "/scan",
        )
        self.assertEqual(
            laser_sensor.find("plugin/frameName").text,
            "base_laser",
        )

        contact_sensor = root.find(
            "./gazebo[@reference='sway_link']/sensor[@type='contact']"
        )
        self.assertIsNotNone(contact_sensor)
        self.assertEqual(
            contact_sensor.find("contact/collision").text,
            "sway_link_collision",
        )
        self.assertEqual(
            contact_sensor.find("plugin/bumperTopicName").text,
            "/base_contacts",
        )

    def test_planar_contact_mode_removes_tangential_friction(self):
        root = ET.fromstring(render(False, True, True, True, True))
        for link_name in ["sway_link"] + [
                "link{}".format(index) for index in range(1, 7)]:
            gazebo_entries = root.findall(
                "./gazebo[@reference='{}']".format(link_name)
            )
            friction_entries = [
                entry for entry in gazebo_entries
                if entry.find("mu1") is not None
            ]
            self.assertEqual(len(friction_entries), 1)
            self.assertEqual(friction_entries[0].find("mu1").text, "0.0")
            self.assertEqual(friction_entries[0].find("mu2").text, "0.0")

    def test_base_only_mode_can_remove_arm_collisions(self):
        regular = ET.fromstring(render(False))
        base_only = ET.fromstring(
            render(False, disable_arm_collisions=True)
        )
        for index in range(1, 7):
            link_name = "link{}".format(index)
            self.assertIsNotNone(
                regular.find("./link[@name='{}']/collision".format(link_name))
            )
            self.assertIsNone(
                base_only.find("./link[@name='{}']/collision".format(link_name))
            )
        self.assertIsNotNone(
            base_only.find("./link[@name='sway_link']/collision")
        )

    def test_arm_joint_limits_match_runtime_safety_contract(self):
        root = ET.fromstring(render(False))
        joints = dict(
            (joint.get("name"), joint)
            for joint in root.findall("joint")
        )
        expected = {
            "joint1": ("-1.57", "1.57"),
            "joint2": ("-1.45", "0.0"),
            "joint3": ("0.0", "0.15"),
            "joint4": ("-1.57", "1.57"),
            "joint5": ("0.0", "0.15"),
            "joint6": ("-1.57", "1.57"),
        }
        for name, bounds in expected.items():
            self.assertNotEqual(joints[name].get("type"), "fixed")
            limit = joints[name].find("limit")
            self.assertEqual(limit.get("lower"), bounds[0])
            self.assertEqual(limit.get("upper"), bounds[1])


if __name__ == "__main__":
    unittest.main()

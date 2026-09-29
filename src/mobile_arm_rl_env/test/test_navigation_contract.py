#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import re
import unittest
import xml.etree.ElementTree as ET


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(PACKAGE_DIR, "config", "navigation")
NAVIGATION_LAUNCH = os.path.join(
    PACKAGE_DIR,
    "launch",
    "planar_base_dwa_navigation.launch",
)
DEMO_LAUNCH = os.path.join(
    PACKAGE_DIR,
    "launch",
    "planar_base_dwa_demo.launch",
)
TEACHER_TRAINING_LAUNCH = os.path.join(
    PACKAGE_DIR,
    "launch",
    "planar_base_dwa_teacher_training.launch",
)


def config_text(name):
    with open(os.path.join(CONFIG_DIR, name), "r") as stream:
        return stream.read()


class NavigationContractTest(unittest.TestCase):
    def assert_pattern(self, text, pattern):
        self.assertIsNotNone(
            re.search(pattern, text),
            "pattern not found: {}".format(pattern),
        )

    def test_dwa_action_space_is_nonholonomic(self):
        planner = config_text("dwa_local_planner.yaml")
        self.assert_pattern(planner, r"(?m)^\s*max_vel_y:\s*0\.0\s*$")
        self.assert_pattern(planner, r"(?m)^\s*min_vel_y:\s*0\.0\s*$")
        self.assert_pattern(planner, r"(?m)^\s*vy_samples:\s*1\s*$")
        self.assert_pattern(planner, r"(?m)^\s*max_vel_x:\s*0\.05\s*$")
        self.assert_pattern(
            planner,
            r"(?m)^\s*min_vel_trans:\s*0\.02\s*$",
        )
        self.assert_pattern(
            planner,
            r"(?m)^\s*max_vel_theta:\s*0\.125\s*$",
        )
        self.assert_pattern(
            planner,
            r"(?m)^\s*min_vel_theta:\s*0\.04\s*$",
        )

    def test_goal_acceptance_matches_measured_stopping_response(self):
        planner = config_text("dwa_local_planner.yaml")
        self.assert_pattern(
            planner,
            r"(?m)^\s*xy_goal_tolerance:\s*0\.08\s*$",
        )
        self.assert_pattern(
            planner,
            r"(?m)^\s*yaw_goal_tolerance:\s*0\.20\s*$",
        )
        self.assert_pattern(
            planner,
            r"(?m)^\s*trans_stopped_vel:\s*0\.01\s*$",
        )
        self.assert_pattern(
            planner,
            r"(?m)^\s*theta_stopped_vel:\s*0\.01\s*$",
        )
        self.assert_pattern(
            planner,
            r"(?m)^\s*latch_xy_goal_tolerance:\s*true\s*$",
        )

    def test_costmaps_use_scan_and_standard_body_frame(self):
        common = config_text("costmap_common.yaml")
        local = config_text("local_costmap.yaml")
        global_costmap = config_text("global_costmap.yaml")
        for costmap in (local, global_costmap):
            self.assertIn("robot_base_frame: base_footprint", costmap)
            self.assertIn("global_frame: base_link", costmap)
            self.assertIn("rolling_window: true", costmap)
        self.assert_pattern(common, r"(?m)^\s*topic:\s*/scan\s*$")
        self.assert_pattern(common, r"(?m)^\s*data_type:\s*LaserScan\s*$")
        self.assert_pattern(common, r"(?m)^\s*marking:\s*true\s*$")
        self.assert_pattern(common, r"(?m)^\s*clearing:\s*true\s*$")
        self.assertEqual(
            len(re.findall(r"(?m)^\s*-\s*\[[^\]]+\]\s*$", common)),
            4,
        )

    def test_move_base_uses_installed_dwa_plugin(self):
        move_base = config_text("move_base.yaml")
        self.assertIn(
            'base_local_planner: "dwa_local_planner/DWAPlannerROS"',
            move_base,
        )
        self.assertIn("recovery_behavior_enabled: false", move_base)
        self.assertIn("oscillation_timeout: 0.0", move_base)

    def test_navigation_launch_owns_only_planar_controllers(self):
        root = ET.parse(NAVIGATION_LAUNCH).getroot()
        nodes = dict(
            (node.get("name"), node)
            for node in root.findall("node")
        )
        self.assertEqual(
            nodes["planar_base_odom"].get("type"),
            "planar_base_odom.py",
        )
        self.assertEqual(
            nodes["planar_cmd_vel_bridge"].get("type"),
            "planar_cmd_vel_bridge.py",
        )
        self.assertEqual(
            nodes["planar_cmd_vel_bridge"].get("if"),
            "$(arg enable_cmd_vel_bridge)",
        )
        self.assertEqual(nodes["move_base"].get("pkg"), "move_base")
        remaps = dict(
            (remap.get("from"), remap.get("to"))
            for remap in nodes["move_base"].findall("remap")
        )
        self.assertEqual(remaps["cmd_vel"], "$(arg cmd_vel_topic)")

    def test_demo_launch_combines_gazebo_and_navigation(self):
        root = ET.parse(DEMO_LAUNCH).getroot()
        includes = [
            include.get("file") for include in root.findall("include")
        ]
        self.assertIn(
            "$(find mobile_arm_description)/launch/"
            "gazebo_planar_avoidance.launch",
            includes,
        )
        self.assertIn(
            "$(find mobile_arm_rl_env)/launch/"
            "planar_base_dwa_navigation.launch",
            includes,
        )

    def test_teacher_training_launch_does_not_connect_dwa_to_base(self):
        root = ET.parse(TEACHER_TRAINING_LAUNCH).getroot()
        navigation_include = root.find(
            "./include[@file='$(find mobile_arm_rl_env)/launch/"
            "planar_base_dwa_navigation.launch']"
        )
        self.assertIsNotNone(navigation_include)
        arguments = dict(
            (argument.get("name"), argument.get("value"))
            for argument in navigation_include.findall("arg")
        )
        self.assertEqual(arguments["cmd_vel_topic"], "/dwa_teacher_cmd_vel")
        self.assertEqual(arguments["enable_cmd_vel_bridge"], "false")
        server = root.find(
            "./node[@type='planar_dwa_env_server.py']"
        )
        self.assertIsNotNone(server)
        parameters = dict(
            (parameter.get("name"), parameter.get("value"))
            for parameter in server.findall("param")
        )
        self.assertEqual(parameters["base_action_scale"], "0.25")
        self.assertEqual(parameters["base_success_threshold"], "0.08")



if __name__ == "__main__":
    unittest.main()

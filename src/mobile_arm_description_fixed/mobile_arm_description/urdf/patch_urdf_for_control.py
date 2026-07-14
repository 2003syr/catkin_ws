#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

urdf_path = os.path.expanduser(
    "~/catkin_ws/src/mobile_arm_description_fixed/mobile_arm_description/urdf/mobile_arm.urdf"
)

joint_limits = {
    "x": (-2.0, 2.0, 100.0, 0.20),
    "y": (-2.0, 2.0, 100.0, 0.20),
    "z": (-3.14, 3.14, 100.0, 0.50),
    "sway": (-1.57, 1.57, 100.0, 0.50),

    "joint1": (-3.14, 3.14, 50.0, 0.50),
    "joint2": (-1.57, 1.57, 50.0, 0.50),
    "joint3": (-0.20, 0.20, 100.0, 0.03),
    "joint4": (-3.14, 3.14, 50.0, 0.50),
    "joint5": (-0.20, 0.20, 100.0, 0.03),
    "joint6": (-3.14, 3.14, 50.0, 0.50),
}

tree = ET.parse(urdf_path)
root = tree.getroot()

# 1. 修改 joint limit
for joint in root.findall("joint"):
    name = joint.get("name")

    if name in joint_limits:
        lower, upper, effort, velocity = joint_limits[name]

        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")

        limit.set("lower", str(lower))
        limit.set("upper", str(upper))
        limit.set("effort", str(effort))
        limit.set("velocity", str(velocity))

# 2. 删除已有 transmission，避免重复添加
for trans in list(root.findall("transmission")):
    root.remove(trans)

# 3. 添加 transmission
for joint_name in joint_limits.keys():
    trans = ET.SubElement(root, "transmission")
    trans.set("name", joint_name + "_trans")

    trans_type = ET.SubElement(trans, "type")
    trans_type.text = "transmission_interface/SimpleTransmission"

    joint = ET.SubElement(trans, "joint")
    joint.set("name", joint_name)

    joint_hw = ET.SubElement(joint, "hardwareInterface")
    joint_hw.text = "hardware_interface/VelocityJointInterface"

    actuator = ET.SubElement(trans, "actuator")
    actuator.set("name", joint_name + "_motor")

    actuator_hw = ET.SubElement(actuator, "hardwareInterface")
    actuator_hw.text = "hardware_interface/VelocityJointInterface"

    reduction = ET.SubElement(actuator, "mechanicalReduction")
    reduction.text = "1"

# 4. 删除已有 gazebo_ros_control 插件，避免重复
for gazebo in list(root.findall("gazebo")):
    plugin = gazebo.find("plugin")
    if plugin is not None and plugin.get("name") == "gazebo_ros_control":
        root.remove(gazebo)

# 5. 添加 gazebo_ros_control 插件
gazebo = ET.SubElement(root, "gazebo")
plugin = ET.SubElement(gazebo, "plugin")
plugin.set("name", "gazebo_ros_control")
plugin.set("filename", "libgazebo_ros_control.so")

robot_namespace = ET.SubElement(plugin, "robotNamespace")
robot_namespace.text = "/"

# 6. 保存
rough_string = ET.tostring(root, "utf-8")
parsed = minidom.parseString(rough_string)
pretty = parsed.toprettyxml(indent="  ")

with open(urdf_path, "w", encoding="utf-8") as f:
    f.write(pretty)

print("URDF patched successfully:")
print(urdf_path)

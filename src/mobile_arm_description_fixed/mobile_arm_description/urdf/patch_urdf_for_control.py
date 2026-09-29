#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deprecated guard for the former destructive URDF patch utility.

The old script silently rewrote x/y/z/sway as movable joints and added
transmissions to them. Configuration now lives in mobile_arm.urdf.xacro, so
this command intentionally never writes a file.
"""

from __future__ import print_function

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the migration summary without modifying any file",
    )
    args = parser.parse_args()
    xacro_path = Path(__file__).resolve().with_name("mobile_arm.urdf.xacro")

    print("DEPRECATED: patch_urdf_for_control.py no longer edits the URDF.")
    print("Configuration source: {}".format(xacro_path))
    print("Protected joints: x, y, z, sway")
    print("No backup was created because no file was modified.")
    print("Use gazebo_control.launch arguments for controlled A/B tests:")
    print("  fixed_base_diagnostic:=true|false")
    print("  disable_arm_gravity:=true|false")
    print("  disable_chassis_gravity:=true|false")
    print("  simple_chassis_collision:=true|false")

    if args.dry_run:
        print("Dry run complete: zero changes.")
        return 0
    print("Refusing to run without --dry-run; destructive patching is disabled.")
    return 2


if __name__ == "__main__":
    sys.exit(main())

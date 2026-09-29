#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""KDL kinematics adapter for rule-based control and RL goal sampling."""

import numpy as np
import PyKDL

from kdl_parser_py.urdf import treeFromParam


class KDLJacobianProvider(object):
    """Expose position FK/Jacobian with an explicit, verified joint order."""

    def __init__(
            self,
            base_frame,
            ee_frame,
            state_joint_names,
            controlled_joint_names,
            robot_description="/robot_description"):
        ok, tree = treeFromParam(robot_description)
        if not ok:
            raise RuntimeError(
                "failed to build KDL tree from {}".format(robot_description)
            )

        self.chain = tree.getChain(base_frame, ee_frame)
        self.solver = PyKDL.ChainJntToJacSolver(self.chain)
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.state_joint_names = list(state_joint_names)
        self.controlled_joint_names = list(controlled_joint_names)
        self.chain_joint_names = []

        for index in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(index).getJoint()
            # Melodic's Python2 PyKDL binding exposes the fixed joint enum as
            # the reserved word ``None`` rather than ``Joint.Fixed``. The
            # stable cross-version API is the textual type name.
            if joint.getTypeName() != "None":
                self.chain_joint_names.append(joint.getName())

        if len(self.chain_joint_names) != self.chain.getNrOfJoints():
            raise RuntimeError("KDL chain joint enumeration is inconsistent")

        missing_state = [
            name for name in self.chain_joint_names
            if name not in self.state_joint_names
        ]
        missing_controlled = [
            name for name in self.controlled_joint_names
            if name not in self.chain_joint_names
        ]
        if missing_state or missing_controlled:
            raise RuntimeError(
                "KDL joint mismatch: missing_state={} missing_controlled={}".format(
                    missing_state,
                    missing_controlled,
                )
            )

        self._state_indices = [
            self.state_joint_names.index(name)
            for name in self.chain_joint_names
        ]
        self._controlled_indices = [
            self.chain_joint_names.index(name)
            for name in self.controlled_joint_names
        ]

    def position_jacobian(self, joint_positions):
        kdl_positions = self._to_kdl_positions(joint_positions)

        jacobian = PyKDL.Jacobian(self.chain.getNrOfJoints())
        result = self.solver.JntToJac(kdl_positions, jacobian)
        if result < 0:
            raise RuntimeError(
                "KDL Jacobian solver failed with code {}".format(result)
            )

        position = np.empty(
            (3, len(self.controlled_joint_names)),
            dtype=np.float64,
        )
        for row in range(3):
            for output_column, chain_column in enumerate(
                    self._controlled_indices):
                position[row, output_column] = jacobian[row, chain_column]

        if not np.all(np.isfinite(position)):
            raise RuntimeError("KDL position Jacobian contains non-finite values")
        return position

    def end_effector_position(self, joint_positions):
        """Return the end-effector position in the configured base frame."""
        kdl_positions = self._to_kdl_positions(joint_positions)
        frame = PyKDL.Frame()
        result = self.fk_solver.JntToCart(kdl_positions, frame)
        if result < 0:
            raise RuntimeError(
                "KDL forward-kinematics solver failed with code {}".format(
                    result
                )
            )

        position = np.asarray(
            [frame.p[0], frame.p[1], frame.p[2]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(position)):
            raise RuntimeError(
                "KDL end-effector position contains non-finite values"
            )
        return position

    def end_effector_rotation(self, joint_positions):
        """Return the 3x3 end-effector rotation in the configured base frame."""
        kdl_positions = self._to_kdl_positions(joint_positions)
        frame = PyKDL.Frame()
        result = self.fk_solver.JntToCart(kdl_positions, frame)
        if result < 0:
            raise RuntimeError(
                "KDL forward-kinematics solver failed with code {}".format(
                    result
                )
            )
        rotation = np.empty((3, 3), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                rotation[row, column] = frame.M[row, column]
        if not np.all(np.isfinite(rotation)):
            raise RuntimeError(
                "KDL end-effector rotation contains non-finite values"
            )
        return rotation

    def _to_kdl_positions(self, joint_positions):
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if joint_positions.shape != (len(self.state_joint_names),):
            raise ValueError(
                "joint_positions must have shape ({},), got {}".format(
                    len(self.state_joint_names),
                    joint_positions.shape,
                )
            )

        kdl_positions = PyKDL.JntArray(self.chain.getNrOfJoints())
        for chain_index, state_index in enumerate(self._state_indices):
            kdl_positions[chain_index] = float(joint_positions[state_index])
        return kdl_positions

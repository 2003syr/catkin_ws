#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class TaskMode(object):
    """Stable task identifiers shared by rule-based and future RL policies."""

    BASE_APPROACH = 0
    ARM_REACH = 1
    RECOVERY = 2

    NAMES = {
        BASE_APPROACH: "BASE_APPROACH",
        ARM_REACH: "ARM_REACH",
        RECOVERY: "RECOVERY",
    }
    COUNT = 3

    @classmethod
    def validate(cls, mode):
        if mode not in cls.NAMES:
            raise ValueError("unknown task mode: {}".format(mode))


class SubgoalType(object):
    """Discrete intent attached to every joint high-level subgoal.

    The physical high-level action remains the verified six-dimensional
    relative pose command.  This separate attribute tells a conditional low
    policy *why* the pose was issued while preserving the physical 6-D action
    contract.
    """

    DIRECT = 0
    DETOUR = 1
    TERMINAL = 2

    NAMES = {
        DIRECT: "DIRECT",
        DETOUR: "DETOUR",
        TERMINAL: "TERMINAL",
    }
    COUNT = 3

    @classmethod
    def validate(cls, subgoal_type):
        subgoal_type = int(subgoal_type)
        if subgoal_type not in cls.NAMES:
            raise ValueError("unknown subgoal type: {}".format(subgoal_type))
        return subgoal_type

    @classmethod
    def name(cls, subgoal_type):
        return cls.NAMES[cls.validate(subgoal_type)]

    @classmethod
    def one_hot(cls, subgoal_type):
        encoded = np.zeros(cls.COUNT, dtype=np.float32)
        encoded[cls.validate(subgoal_type)] = 1.0
        return encoded


class RouteSide(object):
    """Discrete route option, separate from the three semantic stages.

    ``DIRECT / DETOUR / TERMINAL`` remains the stage contract.  A route side
    is only an option parameter for DETOUR, allowing the policy to represent
    the two valid action modes without averaging them into an unsafe command.
    """

    NONE = 0
    UPPER = 1
    LOWER = 2

    NAMES = {
        NONE: "NONE",
        UPPER: "UPPER",
        LOWER: "LOWER",
    }
    COUNT = 3

    @classmethod
    def validate(cls, route_side):
        route_side = int(route_side)
        if route_side not in cls.NAMES:
            raise ValueError("unknown route side: {}".format(route_side))
        return route_side

    @classmethod
    def name(cls, route_side):
        return cls.NAMES[cls.validate(route_side)]

    @classmethod
    def one_hot(cls, route_side):
        encoded = np.zeros(cls.COUNT, dtype=np.float32)
        encoded[cls.validate(route_side)] = 1.0
        return encoded

    @classmethod
    def from_sign(cls, side):
        side = float(side)
        if side > 0.0:
            return cls.UPPER
        if side < 0.0:
            return cls.LOWER
        return cls.NONE

    @classmethod
    def to_sign(cls, route_side):
        route_side = cls.validate(route_side)
        if route_side == cls.UPPER:
            return 1.0
        if route_side == cls.LOWER:
            return -1.0
        return 0.0


class HighLevelOption(object):
    """The four legal joint high-level decisions.

    Keeping semantic stage and detour side as two independently predicted
    categorical variables creates nine mathematical combinations even though
    only four are valid.  This joint option contract makes invalid pairs
    unrepresentable while retaining ``SubgoalType`` and ``RouteSide`` at the
    ROS/environment boundary.
    """

    DIRECT = 0
    DETOUR_UPPER = 1
    DETOUR_LOWER = 2
    TERMINAL = 3

    NAMES = {
        DIRECT: "DIRECT",
        DETOUR_UPPER: "DETOUR_UPPER",
        DETOUR_LOWER: "DETOUR_LOWER",
        TERMINAL: "TERMINAL",
    }
    COUNT = 4

    TYPE_ROUTE = {
        DIRECT: (SubgoalType.DIRECT, RouteSide.NONE),
        DETOUR_UPPER: (SubgoalType.DETOUR, RouteSide.UPPER),
        DETOUR_LOWER: (SubgoalType.DETOUR, RouteSide.LOWER),
        TERMINAL: (SubgoalType.TERMINAL, RouteSide.NONE),
    }
    FROM_TYPE_ROUTE = dict(
        (value, key) for key, value in TYPE_ROUTE.items()
    )

    @classmethod
    def validate(cls, option):
        option = int(option)
        if option not in cls.NAMES:
            raise ValueError("unknown high-level option: {}".format(option))
        return option

    @classmethod
    def name(cls, option):
        return cls.NAMES[cls.validate(option)]

    @classmethod
    def one_hot(cls, option):
        encoded = np.zeros(cls.COUNT, dtype=np.float32)
        encoded[cls.validate(option)] = 1.0
        return encoded

    @classmethod
    def decode(cls, option):
        return cls.TYPE_ROUTE[cls.validate(option)]

    @classmethod
    def encode(cls, subgoal_type, route_side):
        pair = (
            SubgoalType.validate(subgoal_type),
            RouteSide.validate(route_side),
        )
        if pair not in cls.FROM_TYPE_ROUTE:
            raise ValueError(
                "invalid high-level type/route pair: {}".format(pair)
            )
        return cls.FROM_TYPE_ROUTE[pair]


class HighLevelCommand(object):
    """A task decision plus a relative HRL4IN task-state displacement.

    Subgoal layout:

        [base_x, base_y, base_yaw, ee_x, ee_y, ee_z]

    The high-level output is not a direct actuator command.  At a meta step it
    is added to the current task state to form a fixed ideal next state.  The
    low level then observes the remaining displacement to that state.
    """

    SUBGOAL_DIM = 6
    ACTION_DIM = 10

    def __init__(
            self,
            mode,
            subgoal,
            base_priority,
            arm_priority,
            reason="",
            action_mask=None,
            subgoal_mask=None):
        TaskMode.validate(mode)

        subgoal = np.asarray(subgoal, dtype=np.float32)
        if subgoal.shape != (self.SUBGOAL_DIM,):
            raise ValueError(
                "subgoal must have shape (6,), got {}".format(subgoal.shape)
            )

        self.mode = mode
        self.subgoal = subgoal
        self.base_priority = float(np.clip(base_priority, 0.0, 1.0))
        self.arm_priority = float(np.clip(arm_priority, 0.0, 1.0))
        self.reason = str(reason)

        default_action_mask, default_subgoal_mask = self._default_masks(mode)
        self.action_mask = self._mask(
            default_action_mask if action_mask is None else action_mask,
            self.ACTION_DIM,
            "action_mask",
        )
        self.subgoal_mask = self._mask(
            default_subgoal_mask if subgoal_mask is None else subgoal_mask,
            self.SUBGOAL_DIM,
            "subgoal_mask",
        )

    @property
    def mode_name(self):
        return TaskMode.NAMES[self.mode]

    def mode_one_hot(self):
        encoded = np.zeros(TaskMode.COUNT, dtype=np.float32)
        encoded[self.mode] = 1.0
        return encoded

    @classmethod
    def _default_masks(cls, mode):
        """Return HRL4IN execution and task-state masks.

        The first four actions retain the stable virtual-joint layout and the
        final six actions control the physical arm.  In the verified first
        planar stage only x/y are controllable; z and sway remain fixed.  The
        first three subgoal dimensions describe planar-base state and the
        final three describe end-effector position.  Recovery has no task
        subgoal but keeps arm actions enabled to move away from joint limits.
        """
        action_mask = np.zeros(cls.ACTION_DIM, dtype=np.float32)
        subgoal_mask = np.zeros(cls.SUBGOAL_DIM, dtype=np.float32)

        if mode == TaskMode.BASE_APPROACH:
            action_mask[0:2] = 1.0
            subgoal_mask[0:2] = 1.0
        elif mode == TaskMode.ARM_REACH:
            action_mask[4:10] = 1.0
            subgoal_mask[3:6] = 1.0
        elif mode == TaskMode.RECOVERY:
            action_mask[4:10] = 1.0

        return action_mask, subgoal_mask

    @staticmethod
    def _mask(value, expected_size, name):
        mask = np.asarray(value, dtype=np.float32)
        if mask.shape != (expected_size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    expected_size,
                    mask.shape,
                )
            )
        if not np.all(np.isfinite(mask)):
            raise ValueError("{} contains non-finite values".format(name))
        return np.clip(mask, 0.0, 1.0)


class JointHighLevelCommand(object):
    """Six-dimensional joint base/end-effector subgoal command.

    This is the high-level action contract for the fused 66-D/8-D low-level
    controller.  The command is relative to the robot state at the start of a
    high-level period.  ``SubgoalConverter`` immediately freezes it in the
    world frame so it cannot drift with ``base_link``.

    Layout:

        [base_dx, base_dy, base_dyaw, ee_dx, ee_dy, ee_dz]

    The x/y components are expressed in the tracked-base body frame.  The
    end-effector z displacement is parallel to world z.
    """

    SUBGOAL_DIM = 6
    LOW_ACTION_DIM = 8

    DEFAULT_LIMITS = np.asarray(
        [0.50, 0.50, 0.50 * np.pi, 0.15, 0.15, 0.15],
        dtype=np.float32,
    )

    def __init__(
            self,
            base_goal,
            ee_goal,
            reason="",
            normalized_action=None,
            subgoal_type=SubgoalType.DIRECT,
            route_side=RouteSide.NONE):
        self.base_goal = self._vector(base_goal, 3, "base_goal")
        self.ee_goal = self._vector(ee_goal, 3, "ee_goal")
        self.reason = str(reason)
        self.subgoal_type = SubgoalType.validate(subgoal_type)
        self.route_side = RouteSide.validate(route_side)
        if (
                self.subgoal_type != SubgoalType.DETOUR
                and self.route_side != RouteSide.NONE):
            raise ValueError(
                "route_side must be NONE outside the DETOUR stage"
            )
        self.subgoal = np.concatenate((
            self.base_goal,
            self.ee_goal,
        )).astype(np.float32)
        self.action_mask = np.ones(
            self.LOW_ACTION_DIM,
            dtype=np.float32,
        )
        self.subgoal_mask = np.ones(
            self.SUBGOAL_DIM,
            dtype=np.float32,
        )
        if normalized_action is None:
            self.normalized_action = None
        else:
            self.normalized_action = self._vector(
                normalized_action,
                self.SUBGOAL_DIM,
                "normalized_action",
            )

    @property
    def subgoal_type_name(self):
        return SubgoalType.name(self.subgoal_type)

    def subgoal_type_one_hot(self):
        return SubgoalType.one_hot(self.subgoal_type)

    @property
    def route_side_name(self):
        return RouteSide.name(self.route_side)

    def route_side_one_hot(self):
        return RouteSide.one_hot(self.route_side)

    @property
    def option(self):
        return HighLevelOption.encode(
            self.subgoal_type,
            self.route_side,
        )

    @property
    def option_name(self):
        return HighLevelOption.name(self.option)

    def option_one_hot(self):
        return HighLevelOption.one_hot(self.option)

    @classmethod
    def from_action(
            cls,
            action,
            limits=None,
            reason="learned_high_policy",
            subgoal_type=SubgoalType.DIRECT,
            route_side=RouteSide.NONE):
        """Decode a normalized [-1, 1] high-level action."""
        normalized = cls._vector(action, cls.SUBGOAL_DIM, "action")
        normalized = np.clip(normalized, -1.0, 1.0)
        limits = cls._limits(limits)
        physical = normalized * limits
        return cls(
            physical[0:3],
            physical[3:6],
            reason=reason,
            normalized_action=normalized,
            subgoal_type=subgoal_type,
            route_side=route_side,
        )

    def to_action(self, limits=None):
        """Encode the physical command as a normalized PPO action."""
        limits = self._limits(limits)
        return np.clip(
            self.subgoal / limits,
            -1.0,
            1.0,
        ).astype(np.float32)

    @classmethod
    def _limits(cls, limits):
        result = np.asarray(
            cls.DEFAULT_LIMITS if limits is None else limits,
            dtype=np.float32,
        )
        if result.shape != (cls.SUBGOAL_DIM,):
            raise ValueError(
                "limits must have shape (6,), got {}".format(result.shape)
            )
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError("limits must contain finite positive values")
        return result

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    size,
                    vector.shape,
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values".format(name))
        return vector.copy()

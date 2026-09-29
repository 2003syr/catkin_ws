#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load, split, and budget replayable fused-reach scenarios."""

import json

import numpy as np


VALID_SPLITS = ("all", "train", "validation", "test")
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)
DEFAULT_CONTROL_DT = 0.10
DEFAULT_BASE_SPEED = 0.05
DEFAULT_YAW_SPEED = 0.125
DEFAULT_ARM_JOINT_SPEEDS = np.asarray(
    [0.50, 0.50, 0.03, 0.50, 0.03, 0.50],
    dtype=np.float64,
)


def load_fused_reach_scenarios(path):
    with open(path, "r") as stream:
        payload = json.load(stream)
    if int(payload.get("schema_version", -1)) not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported fused reach scenario schema")
    budget = int(payload.get("recommended_step_budget", 0))
    if budget <= 0:
        raise ValueError("scenario set has no positive recommended budget")
    records = payload.get("records", [])
    if int(payload.get("schema_version", -1)) == 3:
        if payload.get("generator") is not None:
            if records:
                raise ValueError(
                    "schema-3 scenario set cannot mix records and generator"
                )
            records = _generate_box_subgoal_records(payload["generator"])
        scenarios = [
            _load_box_subgoal_record(record, index)
            for index, record in enumerate(records)
        ]
    else:
        scenarios = []
        for index, record in enumerate(records):
            scenarios.append(_load_legacy_record(record, index))
    if not scenarios:
        raise ValueError("scenario set contains no records")
    return {
        "path": path,
        "recommended_step_budget": budget,
        "scenarios": scenarios,
        "schema_version": int(payload.get("schema_version", 1)),
        "budget_mode": str(payload.get("budget_mode", "fixed")),
        "recommended_budget_multiplier": payload.get(
            "recommended_budget_multiplier"
        ),
    }


def _load_legacy_record(record, index):
    """Load the original probe/replay record without changing its schema."""
    reset_info = dict(record.get("reset_info", {}))
    target_arm = _vector(
        reset_info.get("target_arm_positions"),
        6,
        "target_arm_positions",
    )
    displacement = _vector(
        reset_info.get("base_displacement_body"),
        2,
        "base_displacement_body",
    )
    target_position = _vector(
        reset_info.get("target_position"),
        3,
        "target_position",
    )
    base_goal_xy = _vector(
        reset_info.get("base_goal_xy"),
        2,
        "base_goal_xy",
    )
    initial_arm = _optional_vector(
        reset_info.get("initial_arm_positions"),
        6,
    )
    lower_bound = record.get("kinematic_lower_bound_steps")
    if lower_bound is None and initial_arm is not None:
        lower_bound = kinematic_lower_bound_steps(
            displacement,
            initial_arm,
            target_arm,
        )
    return {
        "scenario_id": int(record.get("episode", index + 1)),
        "category": str(record.get("category", "unknown")),
        "target_arm_positions": target_arm.tolist(),
        "base_displacement_body": displacement.tolist(),
        "target_position": target_position.tolist(),
        "base_goal_xy": base_goal_xy.tolist(),
        "probe_success_step": record.get("success_step"),
        "probe_initial_distance": float(
            record.get("initial_distance", float("nan"))
        ),
        "probe_final_distance": float(
            record.get("final_distance", float("nan"))
        ),
        "initial_arm_positions": (
            None if initial_arm is None else initial_arm.tolist()
        ),
        "kinematic_lower_bound_steps": (
            None if lower_bound is None else int(lower_bound)
        ),
        "step_budget": (
            None if record.get("step_budget") is None
            else int(record.get("step_budget"))
        ),
        "budget_multiplier": (
            None if record.get("budget_multiplier") is None
            else float(record.get("budget_multiplier"))
        ),
    }


def _load_box_subgoal_record(record, index):
    """Validate and preserve a replayable single-box subgoal scenario."""
    source = dict(record.get("scenario", {}))
    if not source:
        source = dict(record)
    scenario = {
        "scenario_id": int(record.get(
            "episode", source.get("scenario_id", index + 1)
        )),
        "category": str(record.get(
            "category", source.get("category", "box_detour")
        )),
    }
    vector_fields = {
        "initial_base_positions": 3,
        "target_arm_positions": 6,
        "target_position": 3,
        "final_base_goal_xy": 2,
        "box_center_xy": 2,
        "box_size_xy": 2,
    }
    for name, size in vector_fields.items():
        if source.get(name) is not None:
            scenario[name] = _vector(source[name], size, name).tolist()
    scalar_fields = (
        "detour_side",
        "final_base_yaw",
        "base_half_length",
        "base_half_width",
        "path_clearance",
        "waypoint_tolerance",
        "final_base_position_tolerance",
        "final_base_yaw_tolerance",
        "budget_multiplier",
    )
    for name in scalar_fields:
        if source.get(name) is not None:
            value = float(source[name])
            if not np.isfinite(value):
                raise ValueError("{} must be finite".format(name))
            scenario[name] = value
    if source.get("step_budget") is not None:
        scenario["step_budget"] = int(source["step_budget"])
        if scenario["step_budget"] <= 0:
            raise ValueError("step_budget must be positive")
    if source.get("no_obstacle") is not None:
        scenario["no_obstacle"] = bool(source["no_obstacle"])
    return scenario


def _generate_box_subgoal_records(configuration):
    """Expand a compact deterministic single-obstacle curriculum."""
    configuration = dict(configuration)
    if str(configuration.get("type", "")) != "single_box_subgoal_v1":
        raise ValueError("unsupported schema-3 scenario generator")
    count = int(configuration.get("count", 0))
    if count < 10:
        raise ValueError("single-box generator requires at least 10 scenarios")
    no_obstacle_fraction = float(configuration.get(
        "no_obstacle_fraction", 0.20
    ))
    if not 0.0 <= no_obstacle_fraction <= 1.0:
        raise ValueError("no_obstacle_fraction must be in [0, 1]")
    random_state = np.random.RandomState(int(configuration.get("seed", 123)))

    initial_x = _range(configuration, "initial_base_x_range", (-0.03, 0.03))
    initial_y = _range(configuration, "initial_base_y_range", (-0.04, 0.04))
    initial_yaw = _range(
        configuration,
        "initial_base_yaw_range",
        (1.4708, 1.6708),
    )
    box_x = _range(configuration, "box_center_x_range", (0.70, 0.84))
    box_y = _range(configuration, "box_center_y_range", (-0.23, -0.03))
    goal_y = _range(configuration, "final_goal_y_range", (-0.25, 0.05))
    goal_margin = _range(
        configuration, "final_goal_x_margin_range", (0.03, 0.09)
    )
    direct_goal_x = _range(
        configuration, "direct_goal_x_range", (1.20, 1.70)
    )
    safe_base_x = _range(
        configuration, "safe_base_x_range", (-1.75, 1.75)
    )
    safe_base_y = _range(
        configuration, "safe_base_y_range", (-1.75, 1.75)
    )
    final_yaw_offset = _range(
        configuration, "final_yaw_offset_range", (-0.12, 0.12)
    )
    box_size = _vector(
        configuration.get("box_size_xy", [0.20, 0.32]),
        2,
        "box_size_xy",
    )
    base_half_length = float(configuration.get("base_half_length", 0.56))
    base_half_width = float(configuration.get("base_half_width", 0.10))
    path_clearance = float(configuration.get("path_clearance", 0.15))
    if (
            base_half_length <= 0.0
            or base_half_width <= 0.0
            or path_clearance < 0.0):
        raise ValueError("invalid generated route geometry")

    # The planar base uses finite virtual x/y joints.  Reject generator
    # ranges that place a goal or detour inside their safety-filter band;
    # otherwise the forward-only tracker can alternate between DRIVE and
    # ALIGN until timeout even though its geometric route is valid.
    _require_range_inside(initial_x, safe_base_x, "initial_base_x_range")
    _require_range_inside(initial_y, safe_base_y, "initial_base_y_range")
    _require_range_inside(direct_goal_x, safe_base_x, "direct_goal_x_range")
    _require_range_inside(goal_y, safe_base_y, "final_goal_y_range")
    detour_goal_x = (
        box_x[0] + 0.5 * box_size[0] + base_half_length
        + path_clearance + goal_margin[0],
        box_x[1] + 0.5 * box_size[0] + base_half_length
        + path_clearance + goal_margin[1],
    )
    _require_range_inside(
        detour_goal_x, safe_base_x, "generated_detour_goal_x_range"
    )
    lateral_inflation = (
        0.5 * box_size[1] + base_half_width + path_clearance
    )
    detour_path_y = (
        box_y[0] - lateral_inflation,
        box_y[1] + lateral_inflation,
    )
    _require_range_inside(
        detour_path_y, safe_base_y, "generated_detour_path_y_range"
    )

    direct_count = int(round(float(count) * no_obstacle_fraction))
    direct_flags = np.zeros(count, dtype=np.bool_)
    if direct_count > 0:
        direct_flags[random_state.permutation(count)[:direct_count]] = True
    detour_index = 0
    records = []
    for index in range(count):
        start = np.asarray([
            random_state.uniform(*initial_x),
            random_state.uniform(*initial_y),
            random_state.uniform(*initial_yaw),
        ], dtype=np.float64)
        final_yaw = float(
            start[2] + random_state.uniform(*final_yaw_offset)
        )
        no_obstacle = bool(direct_flags[index])
        scenario = {
            "initial_base_positions": start.round(6).tolist(),
            "final_base_yaw": round(final_yaw, 6),
            "no_obstacle": no_obstacle,
            "box_size_xy": box_size.tolist(),
            "path_clearance": path_clearance,
        }
        if no_obstacle:
            scenario["final_base_goal_xy"] = [
                round(random_state.uniform(*direct_goal_x), 6),
                round(random_state.uniform(*goal_y), 6),
            ]
            category = "direct_clear"
        else:
            center = np.asarray([
                random_state.uniform(*box_x),
                random_state.uniform(*box_y),
            ], dtype=np.float64)
            # The terminal x is generated beyond the footprint-inflated box,
            # so BoxDetourPath never has to promote an unsafe requested goal.
            after_x = (
                center[0] + 0.5 * box_size[0]
                + base_half_length + path_clearance
            )
            scenario.update({
                "box_center_xy": center.round(6).tolist(),
                "final_base_goal_xy": [
                    round(after_x + random_state.uniform(*goal_margin), 6),
                    round(random_state.uniform(*goal_y), 6),
                ],
                "detour_side": 1.0 if detour_index % 2 == 0 else -1.0,
            })
            detour_index += 1
            category = "box_detour"
        records.append({
            "episode": index + 1,
            "category": category,
            "scenario": scenario,
        })
    return records


def _range(configuration, name, default):
    values = _vector(configuration.get(name, default), 2, name)
    if values[0] > values[1]:
        raise ValueError("{} lower bound exceeds upper bound".format(name))
    return float(values[0]), float(values[1])


def _require_range_inside(values, bounds, name):
    values = tuple(float(value) for value in values)
    bounds = tuple(float(value) for value in bounds)
    if values[0] < bounds[0] or values[1] > bounds[1]:
        raise ValueError(
            "{}={} leaves safe workspace {}".format(
                name, values, bounds
            )
        )


def kinematic_lower_bound_steps(
        base_displacement_body,
        initial_arm_positions,
        target_arm_positions,
        control_dt=DEFAULT_CONTROL_DT,
        base_speed=DEFAULT_BASE_SPEED,
        yaw_speed=DEFAULT_YAW_SPEED,
        arm_joint_speeds=None):
    """Return a conservative simultaneous-motion lower bound in steps.

    The base translation and arm joints can move concurrently, so the bound
    is the maximum component time.  It is a feasibility normalization, not a
    claim that the closed-loop controller can attain the actuator limits.
    """
    displacement = _vector(
        base_displacement_body, 2, "base_displacement_body"
    )
    initial_arm = _vector(
        initial_arm_positions, 6, "initial_arm_positions"
    )
    target_arm = _vector(
        target_arm_positions, 6, "target_arm_positions"
    )
    control_dt = float(control_dt)
    base_speed = float(base_speed)
    yaw_speed = float(yaw_speed)
    if control_dt <= 0.0 or base_speed <= 0.0 or yaw_speed <= 0.0:
        raise ValueError("kinematic budget parameters must be positive")
    if arm_joint_speeds is None:
        arm_joint_speeds = DEFAULT_ARM_JOINT_SPEEDS
    arm_joint_speeds = _vector(
        arm_joint_speeds, 6, "arm_joint_speeds"
    )
    base_steps = np.linalg.norm(displacement) / (base_speed * control_dt)
    # No target yaw is stored by this task; the translation bound remains
    # valid for a unicycle because Euclidean distance is a lower bound.
    arm_delta = target_arm - initial_arm
    for index in (0, 1, 3, 5):
        arm_delta[index] = np.arctan2(
            np.sin(arm_delta[index]), np.cos(arm_delta[index])
        )
    arm_steps = np.max(
        np.abs(arm_delta) / (arm_joint_speeds * control_dt)
    )
    return int(max(1, np.ceil(max(base_steps, arm_steps))))


def scenario_step_budget(scenario, fallback):
    """Read a per-scenario budget, falling back for schema-1 records."""
    value = scenario.get("step_budget")
    if value is None:
        value = fallback
    value = int(value)
    if value <= 0:
        raise ValueError("scenario step budget must be positive")
    return value


def split_fused_reach_scenarios(
        scenarios,
        split="all",
        validation_fraction=0.10,
        test_fraction=0.20,
        seed=123):
    if split not in VALID_SPLITS:
        raise ValueError("unknown scenario split: {}".format(split))
    scenarios = [dict(item) for item in scenarios]
    if split == "all":
        return scenarios
    if validation_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("scenario split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validation_fraction + test_fraction must be below one"
        )

    random_state = np.random.RandomState(int(seed))
    buckets = {}
    for scenario in scenarios:
        buckets.setdefault(scenario.get("category", "unknown"), []).append(
            scenario
        )
    selected = []
    for category in sorted(buckets):
        bucket = buckets[category]
        order = random_state.permutation(len(bucket)).tolist()
        shuffled = [bucket[index] for index in order]
        validation_count, test_count = _split_counts(
            len(shuffled), validation_fraction, test_fraction
        )
        test_items = shuffled[:test_count]
        validation_items = shuffled[
            test_count:test_count + validation_count
        ]
        train_items = shuffled[test_count + validation_count:]
        if split == "train":
            selected.extend(train_items)
        elif split == "validation":
            selected.extend(validation_items)
        else:
            selected.extend(test_items)
    return sorted(selected, key=lambda item: int(item["scenario_id"]))


class FusedReachScenarioSampler(object):
    """Cycle through scenarios with optional category-balanced sampling.

    ``category_fractions`` assigns probability mass to named categories.  Any
    unassigned mass is distributed across the remaining categories in
    proportion to their scenario counts.  Sampling is deterministic for a
    fixed seed and rotates through every category bucket before reusing an
    item.  Omitting the argument preserves the original one-pass-per-epoch
    behavior used by evaluation.
    """

    def __init__(
            self,
            scenarios,
            seed=123,
            shuffle=True,
            category_fractions=None):
        self.scenarios = [dict(item) for item in scenarios]
        if not self.scenarios:
            raise ValueError("scenario sampler requires at least one task")
        self.random_state = np.random.RandomState(int(seed))
        self.shuffle = bool(shuffle)
        self.category_indices = {}
        for index, scenario in enumerate(self.scenarios):
            category = str(scenario.get("category", "unknown"))
            self.category_indices.setdefault(category, []).append(index)
        self.category_probabilities = _category_probabilities(
            self.category_indices,
            category_fractions,
        )
        self.category_orders = {}
        self.category_cursors = {}
        self.order = []
        self.cursor = 0
        self.epoch = 0
        self._start_epoch()

    def next(self):
        if self.cursor >= len(self.order):
            self._start_epoch()
        scenario = dict(self.scenarios[self.order[self.cursor]])
        self.cursor += 1
        return scenario

    def _start_epoch(self):
        if self.category_probabilities is not None:
            counts = _allocate_category_counts(
                len(self.scenarios),
                self.category_probabilities,
            )
            self.order = []
            for category in sorted(counts):
                for _ in range(counts[category]):
                    self.order.append(self._next_category_index(category))
            if self.shuffle:
                permutation = self.random_state.permutation(
                    len(self.order)
                ).tolist()
                self.order = [self.order[index] for index in permutation]
        elif self.shuffle:
            self.order = self.random_state.permutation(
                len(self.scenarios)
            ).tolist()
        else:
            self.order = list(range(len(self.scenarios)))
        self.cursor = 0
        self.epoch += 1

    def _next_category_index(self, category):
        bucket = self.category_indices[category]
        order = self.category_orders.get(category, [])
        cursor = int(self.category_cursors.get(category, 0))
        if cursor >= len(order):
            if self.shuffle:
                order = self.random_state.permutation(len(bucket)).tolist()
            else:
                order = list(range(len(bucket)))
            cursor = 0
        scenario_index = bucket[order[cursor]]
        self.category_orders[category] = order
        self.category_cursors[category] = cursor + 1
        return scenario_index

    def sampling_category_fractions(self):
        if self.category_probabilities is None:
            denominator = float(len(self.scenarios))
            return dict(
                (category, len(indices) / denominator)
                for category, indices in self.category_indices.items()
            )
        return dict(self.category_probabilities)


def scenario_category_counts(scenarios):
    counts = {}
    for scenario in scenarios:
        category = str(scenario.get("category", "unknown"))
        counts[category] = counts.get(category, 0) + 1
    return counts


def filter_scenario_categories(scenarios, categories=None):
    if not categories:
        return [dict(item) for item in scenarios]
    allowed = set(str(category) for category in categories)
    selected = [
        dict(item) for item in scenarios
        if str(item.get("category", "unknown")) in allowed
    ]
    if not selected:
        raise ValueError(
            "scenario category filter selected no tasks: {}".format(
                sorted(allowed)
            )
        )
    return selected


def _category_probabilities(category_indices, requested_fractions):
    if not requested_fractions:
        return None
    requested = dict(
        (str(category), float(fraction))
        for category, fraction in requested_fractions.items()
    )
    for category, fraction in requested.items():
        if category not in category_indices:
            raise ValueError(
                "sampling category is absent: {}".format(category)
            )
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError(
                "invalid sampling fraction for {}: {}".format(
                    category, fraction
                )
            )
    requested_total = float(sum(requested.values()))
    if requested_total > 1.0 + 1.0e-9:
        raise ValueError("sampling category fractions exceed one")

    probabilities = dict(requested)
    remaining_categories = [
        category for category in category_indices
        if category not in requested
    ]
    remaining_mass = max(0.0, 1.0 - requested_total)
    if remaining_categories:
        remaining_count = float(sum(
            len(category_indices[category])
            for category in remaining_categories
        ))
        for category in remaining_categories:
            probabilities[category] = (
                remaining_mass
                * float(len(category_indices[category]))
                / remaining_count
            )
    elif remaining_mass > 1.0e-9:
        requested_count = float(sum(
            len(category_indices[category]) for category in requested
        ))
        for category in requested:
            probabilities[category] += (
                remaining_mass
                * float(len(category_indices[category]))
                / requested_count
            )
    total = float(sum(probabilities.values()))
    if total <= 0.0:
        raise ValueError("sampling category probabilities sum to zero")
    return dict(
        (category, probability / total)
        for category, probability in probabilities.items()
    )


def _allocate_category_counts(size, probabilities):
    raw = dict(
        (category, float(size) * probability)
        for category, probability in probabilities.items()
    )
    counts = dict(
        (category, int(np.floor(value)))
        for category, value in raw.items()
    )
    missing = int(size - sum(counts.values()))
    order = sorted(
        probabilities,
        key=lambda category: (
            -(raw[category] - counts[category]),
            category,
        ),
    )
    for category in order[:missing]:
        counts[category] += 1
    return counts


def _split_counts(size, validation_fraction, test_fraction):
    if size <= 1:
        return 0, 0
    test_count = int(round(size * float(test_fraction)))
    validation_count = int(round(size * float(validation_fraction)))
    if test_fraction > 0.0:
        test_count = max(test_count, 1)
    if validation_fraction > 0.0 and size >= 3:
        validation_count = max(validation_count, 1)
    while test_count + validation_count >= size:
        if validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    return validation_count, test_count


def _vector(value, size, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(
            "{} must have shape ({},), got {}".format(
                name, size, vector.shape
            )
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} contains non-finite values".format(name))
    return vector


def _optional_vector(value, size):
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        return None
    return vector

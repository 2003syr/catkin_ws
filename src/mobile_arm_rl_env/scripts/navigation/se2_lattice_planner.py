#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Footprint-aware deterministic SE(2) lattice planner."""

from __future__ import print_function

import heapq
import math

import numpy as np


class SE2LatticePlanner(object):
    """Generate neutral, left, and right nonholonomic path candidates."""

    def __init__(
            self,
            heading_bins=16,
            translation_cells=2,
            allow_reverse=True,
            reverse_cost=2.5,
            rotation_cost_radius=0.30,
            goal_tolerance=0.08,
            footprint_half_length=0.56,
            footprint_half_width=0.10,
            safety_margin=0.08,
            side_detour_offset=0.55,
            side_bias=2.0,
            maximum_expansions=120000):
        self.heading_bins = int(heading_bins)
        self.translation_cells = int(translation_cells)
        self.allow_reverse = bool(allow_reverse)
        self.reverse_cost = float(reverse_cost)
        self.rotation_cost_radius = float(rotation_cost_radius)
        self.goal_tolerance = float(goal_tolerance)
        self.footprint_half_length = float(footprint_half_length)
        self.footprint_half_width = float(footprint_half_width)
        self.safety_margin = float(safety_margin)
        self.side_detour_offset = float(side_detour_offset)
        self.side_bias = float(side_bias)
        self.maximum_expansions = int(maximum_expansions)
        self._validate()

    def plan_candidates(self, occupancy_grid, start_pose, goal_xy):
        start_pose = self._vector(start_pose, 3, "start_pose")
        goal_xy = self._vector(goal_xy, 2, "goal_xy")
        start_cell = occupancy_grid.world_to_grid(start_pose[0:2])
        goal_cell = occupancy_grid.world_to_grid(goal_xy)
        if start_cell is None or goal_cell is None:
            raise RuntimeError("start or goal lies outside the local map")

        collision_layers = occupancy_grid.configuration_obstacles(
            self.heading_bins,
            self.footprint_half_length,
            self.footprint_half_width,
            self.safety_margin,
        )
        start_heading = self.angle_to_bin(start_pose[2])
        # The scan itself proves the current physical pose is valid.  Clear
        # only that exact lattice state to tolerate one-cell range noise.
        collision_layers[
            start_heading,
            start_cell[1],
            start_cell[0],
        ] = False

        candidates = []
        signatures = set()
        for side_preference, name in (
                (0, "shortest"),
                (1, "left"),
                (-1, "right")):
            result = self._search(
                occupancy_grid,
                collision_layers,
                start_cell,
                start_heading,
                goal_cell,
                start_pose[0:2],
                goal_xy,
                side_preference,
            )
            if result is None:
                continue
            states, cost, expansions = result
            path = self._states_to_path(occupancy_grid, states)
            signature = tuple(
                (int(state[0]), int(state[1]), int(state[2]))
                for state in states[::max(1, len(states) // 12)]
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            candidates.append({
                "name": name,
                "side_preference": int(side_preference),
                "path": path,
                "cost": float(cost),
                "length": self.path_length(path),
                "expansions": int(expansions),
            })

        if not candidates:
            raise RuntimeError(
                "SE(2) planner found no full-footprint collision-free path"
            )
        candidates.sort(key=lambda item: (
            item["cost"],
            item["length"],
            abs(item["side_preference"]),
        ))
        return candidates

    def _search(
            self,
            occupancy_grid,
            collision_layers,
            start_cell,
            start_heading,
            goal_cell,
            start_xy,
            goal_xy,
            side_preference):
        start = (start_cell[0], start_cell[1], start_heading)
        queue = []
        counter = 0
        heapq.heappush(queue, (
            self._heuristic(occupancy_grid, start_cell, goal_cell),
            counter,
            start,
        ))
        costs = {start: 0.0}
        parents = {start: None}
        expansions = 0
        goal_state = None
        goal_tolerance_cells = max(
            1,
            int(math.ceil(
                self.goal_tolerance / occupancy_grid.resolution
            )),
        )

        while queue and expansions < self.maximum_expansions:
            estimated_cost, unused_counter, state = heapq.heappop(queue)
            state_cost = costs.get(state)
            if state_cost is None:
                continue
            heuristic = self._heuristic(
                occupancy_grid,
                state[0:2],
                goal_cell,
            )
            if estimated_cost > state_cost + heuristic + 1.0e-9:
                continue
            expansions += 1
            if (
                    abs(state[0] - goal_cell[0]) <= goal_tolerance_cells
                    and abs(state[1] - goal_cell[1])
                    <= goal_tolerance_cells
                    and math.hypot(
                        state[0] - goal_cell[0],
                        state[1] - goal_cell[1],
                    ) <= goal_tolerance_cells):
                goal_state = state
                break

            for next_state, primitive_cost in self._neighbors(
                    state,
                    collision_layers,
                    occupancy_grid.resolution):
                next_world = occupancy_grid.grid_to_world(
                    next_state[0:2]
                )
                biased_cost = primitive_cost + self._side_cost(
                    next_world,
                    start_xy,
                    goal_xy,
                    side_preference,
                    primitive_cost,
                )
                next_cost = state_cost + biased_cost
                if next_cost + 1.0e-9 >= costs.get(
                        next_state, float("inf")):
                    continue
                costs[next_state] = next_cost
                parents[next_state] = state
                counter += 1
                heapq.heappush(queue, (
                    next_cost + self._heuristic(
                        occupancy_grid,
                        next_state[0:2],
                        goal_cell,
                    ),
                    counter,
                    next_state,
                ))

        if goal_state is None:
            return None
        states = []
        state = goal_state
        while state is not None:
            states.append(state)
            state = parents[state]
        states.reverse()
        return states, costs[goal_state], expansions

    def _neighbors(self, state, collision_layers, resolution):
        ix, iy, heading_index = state
        results = []
        angle_step = 2.0 * math.pi / float(self.heading_bins)
        for delta in (-1, 1):
            next_heading = (heading_index + delta) % self.heading_bins
            if collision_layers[next_heading, iy, ix]:
                continue
            results.append((
                (ix, iy, next_heading),
                angle_step * self.rotation_cost_radius,
            ))

        heading = self.bin_to_angle(heading_index)
        dx = int(round(
            self.translation_cells * math.cos(heading)
        ))
        dy = int(round(
            self.translation_cells * math.sin(heading)
        ))
        if dx == 0 and dy == 0:
            return results
        step_length = resolution * math.hypot(dx, dy)
        directions = ((1, 1.0),)
        if self.allow_reverse:
            directions = ((1, 1.0), (-1, self.reverse_cost))
        height = collision_layers.shape[1]
        width = collision_layers.shape[2]
        for direction, multiplier in directions:
            next_x = ix + direction * dx
            next_y = iy + direction * dy
            if (
                    next_x < 0 or next_x >= width
                    or next_y < 0 or next_y >= height):
                continue
            if collision_layers[
                    heading_index, next_y, next_x]:
                continue
            midpoint_x = int(round(0.5 * (ix + next_x)))
            midpoint_y = int(round(0.5 * (iy + next_y)))
            if collision_layers[
                    heading_index, midpoint_y, midpoint_x]:
                continue
            results.append((
                (next_x, next_y, heading_index),
                step_length * multiplier,
            ))
        return results

    def _side_cost(
            self,
            point,
            start_xy,
            goal_xy,
            side_preference,
            primitive_cost):
        if side_preference == 0:
            return 0.0
        goal_vector = goal_xy - start_xy
        goal_length = float(np.linalg.norm(goal_vector))
        if goal_length <= 1.0e-6:
            return 0.0
        unit = goal_vector / goal_length
        relative = point - start_xy
        progress = float(np.clip(
            np.dot(relative, unit) / goal_length,
            0.0,
            1.0,
        ))
        lateral = float(
            unit[0] * relative[1] - unit[1] * relative[0]
        )
        desired = (
            float(side_preference)
            * self.side_detour_offset
            * math.sin(math.pi * progress)
        )
        return (
            self.side_bias
            * abs(lateral - desired)
            * primitive_cost
        )

    def _states_to_path(self, occupancy_grid, states):
        path = np.zeros((len(states), 3), dtype=np.float64)
        for index, state in enumerate(states):
            path[index, 0:2] = occupancy_grid.grid_to_world(
                state[0:2]
            )
            path[index, 2] = self.bin_to_angle(state[2])
        return path

    def angle_to_bin(self, angle):
        wrapped = math.atan2(math.sin(angle), math.cos(angle))
        index = int(round(
            (wrapped + math.pi)
            * float(self.heading_bins)
            / (2.0 * math.pi)
        ))
        return index % self.heading_bins

    def bin_to_angle(self, index):
        return (
            -math.pi
            + 2.0 * math.pi
            * float(int(index) % self.heading_bins)
            / float(self.heading_bins)
        )

    @staticmethod
    def _heuristic(occupancy_grid, cell, goal_cell):
        return occupancy_grid.resolution * math.hypot(
            int(cell[0]) - int(goal_cell[0]),
            int(cell[1]) - int(goal_cell[1]),
        )

    @staticmethod
    def path_length(path):
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(
            np.diff(path[:, 0:2], axis=0),
            axis=1,
        )))

    def _validate(self):
        if self.heading_bins < 8:
            raise ValueError("heading_bins must be at least 8")
        if self.translation_cells <= 0:
            raise ValueError("translation_cells must be positive")
        if self.reverse_cost < 1.0:
            raise ValueError("reverse_cost must be at least one")
        if self.rotation_cost_radius <= 0.0:
            raise ValueError("rotation_cost_radius must be positive")
        if self.goal_tolerance <= 0.0:
            raise ValueError("goal_tolerance must be positive")
        if self.footprint_half_length <= 0.0:
            raise ValueError("footprint_half_length must be positive")
        if self.footprint_half_width <= 0.0:
            raise ValueError("footprint_half_width must be positive")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")
        if self.side_detour_offset <= 0.0 or self.side_bias < 0.0:
            raise ValueError("side candidate parameters are invalid")
        if self.maximum_expansions <= 0:
            raise ValueError("maximum_expansions must be positive")

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                "{} must be a finite shape-{} vector".format(name, size)
            )
        return vector

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


def clamp(x, low, high):
    return max(low, min(high, x))


def angle_wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class Simple2DEnv:
    def __init__(self):
        self.dt = 0.1
        self.max_v = 0.25
        self.max_w = 0.8

        self.world_size = 4.0

        self.goal = np.array([1.5, 1.2])
        self.obstacle = np.array([0.6, 0.5])
        self.obstacle_radius = 0.25

        self.robot = np.zeros(3)  # x, y, yaw
        self.reset()

    def reset(self):
        self.robot = np.array([-1.4, -1.0, 0.0], dtype=np.float32)
        return self.get_state()

    def get_state(self):
        x, y, yaw = self.robot

        dx = self.goal[0] - x
        dy = self.goal[1] - y

        dist_goal = math.sqrt(dx * dx + dy * dy)
        angle_goal = angle_wrap(math.atan2(dy, dx) - yaw)

        ox = self.obstacle[0] - x
        oy = self.obstacle[1] - y
        dist_obs = math.sqrt(ox * ox + oy * oy) - self.obstacle_radius
        angle_obs = angle_wrap(math.atan2(oy, ox) - yaw)

        return np.array([
            dx,
            dy,
            dist_goal,
            angle_goal,
            dist_obs,
            angle_obs,
        ], dtype=np.float32)

    def step_low_level(self, subgoal):
        """
        低层控制器：让机器人朝高层给的局部子目标移动。
        subgoal 是世界坐标系下的二维点 [gx, gy]
        """
        x, y, yaw = self.robot

        gx, gy = subgoal
        dx = gx - x
        dy = gy - y

        target_angle = math.atan2(dy, dx)
        angle_err = angle_wrap(target_angle - yaw)
        dist = math.sqrt(dx * dx + dy * dy)

        v = clamp(0.8 * dist, -self.max_v, self.max_v)
        w = clamp(2.0 * angle_err, -self.max_w, self.max_w)

        # 简单差速运动学
        x = x + v * math.cos(yaw) * self.dt
        y = y + v * math.sin(yaw) * self.dt
        yaw = angle_wrap(yaw + w * self.dt)

        self.robot = np.array([x, y, yaw], dtype=np.float32)

        return self.get_state()

    def check_collision(self):
        x, y, _ = self.robot
        d = np.linalg.norm(np.array([x, y]) - self.obstacle)
        return d < self.obstacle_radius + 0.12

    def check_success(self):
        x, y, _ = self.robot
        d = np.linalg.norm(np.array([x, y]) - self.goal)
        return d < 0.12


class FakeHighLevelPolicy:
    def __init__(self):
        self.max_subgoal_step = 0.45

    def predict(self, robot_xy, goal_xy, obstacle_xy, obstacle_radius):
        """
        假高层：不是神经网络，只是规则。
        作用是验证：
        高层输出局部子目标 → 低层执行 → 目标距离减小
        """
        robot_xy = np.array(robot_xy)
        goal_xy = np.array(goal_xy)
        obstacle_xy = np.array(obstacle_xy)

        direction = goal_xy - robot_xy
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            return goal_xy

        direction = direction / dist

        # 默认朝目标前进一小步
        subgoal = robot_xy + direction * min(self.max_subgoal_step, dist)

        # 如果直线方向上靠近障碍，则绕一下
        to_obs = obstacle_xy - robot_xy
        obs_dist = np.linalg.norm(to_obs)

        if obs_dist < 0.8:
            to_obs_unit = to_obs / max(obs_dist, 1e-6)

            # 如果目标方向和障碍方向夹角很小，说明障碍挡路
            dot = np.dot(direction, to_obs_unit)

            if dot > 0.65:
                # 绕障：取垂直方向
                perp = np.array([-to_obs_unit[1], to_obs_unit[0]])
                subgoal = robot_xy + 0.35 * perp + 0.20 * direction

        return subgoal


def run_demo():
    env = Simple2DEnv()
    high_policy = FakeHighLevelPolicy()

    high_period = 10
    max_steps = 400

    state = env.reset()
    current_subgoal = np.array([env.robot[0], env.robot[1]])

    traj = []
    subgoals = []
    rewards = []
    distances = []
    collisions = []

    prev_dist = np.linalg.norm(env.robot[:2] - env.goal)

    total_reward = 0.0

    for step in range(max_steps):
        robot_xy = env.robot[:2]

        # 高层每隔 high_period 步输出一次子目标
        if step % high_period == 0:
            current_subgoal = high_policy.predict(
                robot_xy=robot_xy,
                goal_xy=env.goal,
                obstacle_xy=env.obstacle,
                obstacle_radius=env.obstacle_radius
            )

        state = env.step_low_level(current_subgoal)

        now_dist = np.linalg.norm(env.robot[:2] - env.goal)
        collision = env.check_collision()
        success = env.check_success()

        # 高层奖励形式：距离缩短 + 成功奖励 - 碰撞惩罚 - 时间惩罚
        reward = 2.0 * (prev_dist - now_dist) - 0.01

        if collision:
            reward -= 3.0

        if success:
            reward += 10.0

        total_reward += reward

        traj.append(env.robot.copy())
        subgoals.append(current_subgoal.copy())
        rewards.append(total_reward)
        distances.append(now_dist)
        collisions.append(collision)

        prev_dist = now_dist

        if success:
            print("Success at step:", step)
            break

        if collision:
            print("Collision at step:", step)
            break

    traj = np.array(traj)
    subgoals = np.array(subgoals)

    print("Total reward:", total_reward)
    print("Final distance:", distances[-1])

    plot_results(env, traj, subgoals, rewards, distances, collisions)


def plot_results(env, traj, subgoals, rewards, distances, collisions):
    # 图 1：轨迹图
    plt.figure(figsize=(7, 7))

    plt.plot(traj[:, 0], traj[:, 1], label="robot trajectory")
    plt.scatter(env.goal[0], env.goal[1], marker="*", s=200, label="goal")
    plt.scatter(traj[0, 0], traj[0, 1], marker="o", s=100, label="start")

    # 障碍物
    circle = plt.Circle(
        env.obstacle,
        env.obstacle_radius,
        fill=False,
        linewidth=2,
        label="obstacle"
    )
    plt.gca().add_patch(circle)

    # 子目标点，隔一段画一个
    if len(subgoals) > 0:
        idx = np.arange(0, len(subgoals), 10)
        plt.scatter(subgoals[idx, 0], subgoals[idx, 1], s=25, label="high-level subgoals")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.title("2D HRL Demo: Trajectory and High-Level Subgoals")
    plt.legend()
    plt.tight_layout()
    plt.savefig("simple_2d_hrl_trajectory.png", dpi=200)

    # 图 2：距离变化
    plt.figure(figsize=(8, 4))
    plt.plot(distances)
    plt.grid(True)
    plt.xlabel("step")
    plt.ylabel("distance to goal / m")
    plt.title("Distance to Goal")
    plt.tight_layout()
    plt.savefig("simple_2d_hrl_distance.png", dpi=200)

    # 图 3：累计奖励
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.grid(True)
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.title("Cumulative Reward")
    plt.tight_layout()
    plt.savefig("simple_2d_hrl_reward.png", dpi=200)

    print("Saved figures:")
    print("  simple_2d_hrl_trajectory.png")
    print("  simple_2d_hrl_distance.png")
    print("  simple_2d_hrl_reward.png")

    plt.show()


if __name__ == "__main__":
    run_demo()

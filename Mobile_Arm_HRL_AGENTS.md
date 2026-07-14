# Mobile Arm Hierarchical Reinforcement Learning Project - AGENTS.md

## 1. Project Overview

This project develops a hierarchical reinforcement learning (HRL)
framework for a mobile manipulator using ROS and Gazebo.

The target platform is:

-   Mobile base
-   Multi-joint robotic arm
-   End-effector mounted sensor (camera hardware exists, but
    camera-based task is currently not the research focus)

The current research focus is:

> Task-driven hierarchical reinforcement learning for mobile
> manipulators.

The goal is not simply to train a robot to reach a target. The goal is
to design a hierarchical decision framework that allows a mobile
manipulator to decompose complex tasks into meaningful sub-tasks and
execute them efficiently.

------------------------------------------------------------------------

# 2. Research Motivation

## Problem

A mobile manipulator has a large action space:

-   Mobile base motion
-   Arm joint motion
-   End-effector positioning
-   Collision avoidance
-   Joint safety constraints

Using a single reinforcement learning policy directly mapping
observations to all robot actions creates problems:

-   Large exploration space
-   Slow convergence
-   Poor interpretability
-   Difficulty transferring between tasks

Current idea:

Use hierarchical reinforcement learning.

Instead of:

    observation
        |
    single RL policy
        |
    robot action

Use:

    observation
        |
    High-level policy
        |
    task decision / sub-goal
        |
    Low-level policy
        |
    continuous control action
        |
    robot

------------------------------------------------------------------------

# 3. Main Research Direction

## Task-driven Hierarchical Reinforcement Learning for Mobile Manipulators

The main innovation direction is:

> A task-driven hierarchical reinforcement learning method that improves
> mobile manipulator decision efficiency by introducing task-level
> decomposition and state-aware sub-goal generation.

The purpose is not to rename existing HRL structures.

The focus is solving practical problems in mobile manipulation:

-   How should high-level policy divide tasks?
-   How should sub-goals be generated?
-   How can high-level decisions improve low-level control efficiency?
-   How can the robot switch between different behaviors?

------------------------------------------------------------------------

# 4. Current Algorithm Concept

The target architecture:

                     Environment

                          |
                          |

                  High-level RL

                          |

              Task selection / Sub-goal

                          |

                  Low-level RL

                          |

                      Action

                          |

                 Safety execution layer

                          |

                      Robot

------------------------------------------------------------------------

# 5. High-Level Policy Design

The high-level policy does not directly output joint commands.

Its responsibility:

-   Determine current task mode
-   Generate sub-goal
-   Decide motion priority

Possible high-level tasks:

    Base approach

    Arm reaching

    Obstacle avoidance

    Recovery behavior

The final task set will be determined during algorithm development.

------------------------------------------------------------------------

# 6. Low-Level Policy Design

The low-level policy executes the high-level command.

Input:

-   Robot state
-   Current sub-goal
-   Environment information

Output:

-   Base velocity
-   Joint velocity
-   Motion command

Current action space:

10 dimensions:

    x
    y
    z
    sway

    joint1
    joint2
    joint3
    joint4
    joint5
    joint6

------------------------------------------------------------------------

# 7. Sub-goal Design Philosophy

Sub-goals should not be arbitrary points.

They should consider robot execution capability.

Potential information:

-   Target relative position
-   End-effector error
-   Base-target distance
-   Joint state
-   Joint limit margin
-   Obstacle information

The project currently treats sub-goal feasibility as an important
mechanism.

However:

The main contribution is not only "feasible sub-goal generation".

The contribution should focus on:

-   Task-driven decision
-   Improved learning efficiency
-   Better behavior decomposition

------------------------------------------------------------------------

# 8. Simulation Environment

## Workspace

    ~/catkin_ws

Build:

``` bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

------------------------------------------------------------------------

# 9. Robot Description Package

Path:

    catkin_ws/src/mobile_arm_description_fixed/mobile_arm_description

Responsible for:

-   Robot URDF
-   Gazebo model
-   Controllers
-   Launch files

------------------------------------------------------------------------

## URDF

File:

    urdf/mobile_arm.urdf

Defines:

-   Robot structure
-   Links
-   Joints
-   TF relationships
-   Joint limits
-   Gazebo properties

Important frames:

    base_link

    x-link
    y-link
    z-link

    sway_link

    link1-link6

Current end-effector reference:

    link6

The camera frame is temporarily ignored.

------------------------------------------------------------------------

# 10. Gazebo Control System

Controller configuration:

    config/mobile_arm_controllers.yaml

Controllers:

    joint_state_controller

    x_velocity_controller
    y_velocity_controller
    z_velocity_controller
    sway_velocity_controller

    joint1_velocity_controller
    joint2_velocity_controller
    joint3_velocity_controller
    joint4_velocity_controller
    joint5_velocity_controller
    joint6_velocity_controller

Launch:

    launch/gazebo_control.launch

Starts:

-   Gazebo
-   robot description
-   robot state publisher
-   controllers

------------------------------------------------------------------------

# 11. Reinforcement Learning Environment

Main file:

    mobile_arm_rl_env/scripts/mobile_arm_env_check.py

Responsibilities:

-   Obtain robot state
-   Construct observation
-   Receive action
-   Convert action to robot command
-   Publish controller commands
-   Calculate reward
-   Apply safety restrictions

------------------------------------------------------------------------

# 12. Observation Design

Current observation dimension:

    46

Structure:

    0-2:
    target position in base frame

    3-5:
    target position in link6 frame

    6-8:
    link6 position in base frame

    9:
    base-target distance

    10:
    end-effector-target distance

    11-20:
    joint positions q

    21-30:
    joint velocities dq

    31-40:
    joint limit margin q_margin

    41-45:
    obstacle information placeholder

Observation design should preserve information required for future HRL.

------------------------------------------------------------------------

# 13. Fixed Joint Order

Never rely on ROS JointState order.

Use:

``` python
[
"x",
"y",
"z",
"sway",
"joint1",
"joint2",
"joint3",
"joint4",
"joint5",
"joint6"
]
```

Observation and action dimensions depend on this order.

------------------------------------------------------------------------

# 14. Action Pipeline

Current control pipeline:

    RL action

    ↓

    decode_action()

    ↓

    velocity command

    ↓

    action safety filter

    ↓

    ROS controller

    ↓

    Gazebo

    ↓

    robot motion

The safety layer must remain after policy output.

------------------------------------------------------------------------

# 15. Safety Layer

Purpose:

Prevent unsafe exploration.

Current information:

    q_margin

represents distance from joint limits.

Rules:

-   Prevent movement toward dangerous joint limits.
-   Allow movement away from limits.
-   Do not replace the RL policy.

Architecture:

    RL policy

    ↓

    Safety filter

    ↓

    Robot

------------------------------------------------------------------------

# 16. Current Completed Work

Completed:

-   Robot URDF loading
-   Gazebo simulation
-   RViz visualization
-   TF verification
-   Joint state acquisition
-   Controller configuration
-   Observation construction
-   Action execution
-   Joint limit margin calculation
-   Action safety filtering

Current system:

    Gazebo

    ↓

    TF + joint_states

    ↓

    RL environment

    ↓

    observation

    ↓

    policy

    ↓

    action

    ↓

    controller

    ↓

    robot motion

------------------------------------------------------------------------

# 17. Future Development Roadmap

## Stage 1: HRL Framework

Implement:

    High-level policy

    ↓

    Low-level policy

    ↓

    Robot

First use rule-based policies to verify framework.

Then replace with RL algorithms.

------------------------------------------------------------------------

## Stage 2: High-Level Algorithm

Research focus:

-   Task decomposition
-   Sub-goal generation
-   Task switching mechanism
-   Learning efficiency improvement

Possible algorithms:

-   PPO
-   SAC
-   HIRO-style HRL
-   Goal-conditioned RL

------------------------------------------------------------------------

## Stage 3: Experimental Evaluation

Compare:

    Single-layer RL

    Standard HRL

    Proposed task-driven HRL

Metrics:

-   Success rate
-   Training convergence speed
-   Episode reward
-   Motion efficiency
-   Constraint violations

------------------------------------------------------------------------

# 18. Development Rules for Codex

When modifying code:

1.  Keep ROS interfaces unchanged unless necessary.

2.  Keep fixed joint ordering.

3.  Keep observation dimension consistent.

4.  Do not remove safety filtering.

5.  Separate:

    -   Environment code
    -   RL algorithm code
    -   Training code

6.  Avoid adding camera-related task assumptions unless explicitly
    requested.

7.  The current research priority is:

```{=html}
<!-- -->
```
    Task-driven HRL
    for mobile manipulator control

not visual servoing or camera tracking.

------------------------------------------------------------------------

# 19. Current Project Status

Current stage:

    Simulation environment completed.

    Robot control loop completed.

    RL environment foundation completed.

    Preparing hierarchical reinforcement learning algorithm design.

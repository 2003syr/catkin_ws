# Mobile Arm HRL Catkin Workspace

ROS Melodic + Gazebo 移动机械臂分层强化学习基础工作空间。

## 主代码

```text
src/mobile_arm_description_fixed/mobile_arm_description
src/mobile_arm_rl_env
```

- `mobile_arm_description`：参数化 Xacro、Gazebo 模型、控制器和空闲诊断。
- `mobile_arm_rl_env`：46 维观测、10 维动作、安全过滤、规则型 HRL 和 Jacobian DLS 低层控制。

固定接口：

```text
observation: 46
action: 10
joint order: x, y, z, sway, joint1, joint2, joint3, joint4, joint5, joint6
```

默认诊断模式把 `x/y/z/sway` 建成真正 fixed joint，但仍在 RL 接口中保留四个零值占位量。

## 编译

```bash
cd ~/vm_code/catkin_ws
catkin_make
source devel/setup.bash
```

## 启动

```bash
# 终端1
roslaunch mobile_arm_description gazebo_control.launch

# 终端2
roslaunch mobile_arm_rl_env rule_based_hrl.launch
```

`gazebo_control.launch` 默认使用已通过空闲稳定性验收的控制链诊断配置：

```text
fixed_base_diagnostic=true
disable_arm_gravity=true
disable_chassis_gravity=false
simple_chassis_collision=false
```

这些参数可用于单变量 A/B 实验。未来启用真实移动底盘时应使用明确的 `/cmd_vel` 或底盘插件，不应把零行程 `x/y/z/sway` 作为正式底盘控制接口。

## 验证

```bash
# 空闲 15 秒采样
rosrun mobile_arm_description diagnose_idle_motion.py \
  --warmup 3 --duration 15 --label manual_check

# 每个机械臂关节正向、反向和归零速度
rosrun mobile_arm_description verify_velocity_response.py

# HRL 纯逻辑测试
python src/mobile_arm_rl_env/test/test_hrl_framework.py

# Xacro、fixed joint 和 KDL Jacobian 测试
python src/mobile_arm_description_fixed/mobile_arm_description/test/test_description_contract.py
```

空闲自转的根因证据、A/B 数据和回滚方法见：

- [GAZEBO_IDLE_MOTION_AB_REPORT.md](GAZEBO_IDLE_MOTION_AB_REPORT.md)
- [GAZEBO_重复底盘修复说明.md](GAZEBO_重复底盘修复说明.md)
- [HRL_低层笛卡尔控制说明.md](HRL_低层笛卡尔控制说明.md)
- [Mobile_Arm_HRL_AGENTS.md](Mobile_Arm_HRL_AGENTS.md)

## Git 与第三方依赖

```bash
git clone --recurse-submodules https://github.com/2003syr/catkin_ws.git
cd catkin_ws
git submodule update --init --recursive
```

`build/`、`devel/`、日志、CSV 和本地压缩包不进入版本控制。

# Mobile Arm HRL Catkin Workspace

ROS + Gazebo 移动机械臂强化学习工作区。仓库保存两个可追溯版本：

- `baseline-v1`：开始 HRL 重构前的原始源码基线。
- `rule-based-hrl-v2`：任务驱动的规则型分层强化学习基础框架。

## 获取源码

第三方 ROS 包以 Git 子模块保存：

```bash
git clone --recurse-submodules https://github.com/2003syr/catkin_ws.git
cd catkin_ws
git submodule update --init --recursive
```

Catkin 编译目录、运行日志、数据文件和本地压缩包不进入版本控制。

## 编译

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 基础环境

基础环境保持以下接口约束：

- observation：固定 46 维。
- action：固定 10 维。
- 关节顺序：`x, y, z, sway, joint1, joint2, joint3, joint4, joint5, joint6`。
- 所有策略动作必须经过关节限位安全过滤后才能发送给控制器。

项目设计背景和开发约束见 [Mobile_Arm_HRL_AGENTS.md](Mobile_Arm_HRL_AGENTS.md)。

## 规则型 HRL v2

第二个版本实现了用于验证接口的任务驱动分层框架：

```text
46维 observation
  -> 高层任务选择（BASE_APPROACH / ARM_REACH / RECOVERY）
  -> 6维局部子目标与底座/机械臂优先级
  -> 规则低层策略
  -> 10维归一化动作
  -> 原有安全过滤
  -> ROS velocity controllers
```

高层默认每 10 个低层控制步更新一次，并使用进入/退出两个距离阈值避免任务模式抖动。规则低层仅用于验证任务切换、时间尺度和安全执行链路，不作为最终逆运动学或学习控制器。

启动方式：

```bash
# 终端 1
source devel/setup.bash
roslaunch mobile_arm_description gazebo_control.launch

# 终端 2
source devel/setup.bash
roslaunch mobile_arm_rl_env rule_based_hrl.launch
```

运行不依赖 ROS/Gazebo 的基础逻辑测试：

```bash
python src/mobile_arm_rl_env/test/test_hrl_framework.py
```

关键实现：

- `scripts/hrl/high_level_command.py`：稳定的任务命令协议。
- `scripts/hrl/rule_based_high_policy.py`：可解释的任务切换和可行子目标裁剪。
- `scripts/hrl/rule_based_low_policy.py`：用于闭环接线验证的低层规则策略。
- `scripts/hrl/hierarchical_env.py`：高低层双时间尺度调度。
- `scripts/run_rule_based_hrl.py`：ROS 运行入口。

当前限制：episode reset 仍只清零控制指令和计数；障碍输入仍是占位值；规则低层不是 Jacobian 逆解。后续应先验证 Gazebo 控制方向，再用目标条件强化学习策略替换规则低层。

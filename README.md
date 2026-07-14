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


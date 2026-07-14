# mobile_arm_description 使用说明

这是从原始“移动机械臂.7z”整理出的 ROS URDF 描述包，主要用于 RViz/Gazebo 可视化。

## 放入工作空间

```bash
cd ~/catkin_ws/src
cp -r /path/to/mobile_arm_description .
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## RViz 查看模型

```bash
roslaunch mobile_arm_description display.launch
```

## Gazebo 加载模型

```bash
roslaunch mobile_arm_description gazebo.launch
```

## 注意

原始 URDF 的所有关节 limit 都是 0，当前只能稳定用于模型显示。若要拖动关节、MoveIt 规划或 Gazebo 控制，需要补充关节上下限、effort、velocity、transmission、controller 配置等。

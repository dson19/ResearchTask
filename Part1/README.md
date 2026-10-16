# Part 1 — ROSOrin Pro Gazebo Simulation

SLAM mapping and autonomous navigation (single goal and multi-point) of the Hiwonder ROSOrin Pro
in Gazebo, based on Chapter 9 of the
[official documentation](https://wiki.hiwonder.com/projects/rosorin-pro/en/latest/docs/9_Gazebo_Simulation.html).

## Repository layout

```
Part1/
├── ros2_ws/                     # ROS 2 workspace
│   ├── .typerc                  # Hiwonder environment configuration
│   └── src/
│       ├── robot_gazebo/        # worlds, launch files, SLAM/Nav2 config, maps/ (map_01 included)
│       └── rosorin_description/ # robot URDF and meshes
└── README.md
```

## Requirements

- Ubuntu 22.04 with ROS 2 Humble.
- Tested in the Hiwonder ROSOrin Pro virtual-machine image (VMware). The VMware installer and the
  VM image can be downloaded from
  [Google Drive](https://drive.google.com/drive/folders/1hd0ENvzHaCynXpfEniMJ1CoeD4yFEFbr?usp=sharing).
  Import the image in VMware with *Open a Virtual Machine*. The image already contains
  Ignition Gazebo, `ros_ign_gazebo`, `ros_ign_bridge`, `slam_toolbox`, Nav2, `xacro` and the
  `holonomic_sim` package required by `room_worlds.launch.py`.
- VMware settings: network adapter *Bridged*, *Accelerate 3D graphics* disabled.

## 1. Get the code and build

```bash
git clone https://github.com/dson19/ResearchTask.git
cd ResearchTask/Part1
export PART1=$(pwd)

cd $PART1/ros2_ws
colcon build --symlink-install
```

## 2. Set up the environment

Load the workspace and the Hiwonder configuration in every new terminal:

```bash
echo "source $PART1/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "source $PART1/ros2_ws/.typerc" >> ~/.bashrc
echo "export PART1=$PART1" >> ~/.bashrc
source ~/.bashrc
```

A new terminal now prints the configuration: `MS200 | aurora | ROSOrin_Mecanum`,
`ROS_DOMAIN_ID: 0`.

## 3. View the robot model (optional)

```bash
ros2 launch robot_gazebo worlds.launch.py
```

The robot appears in an empty world. Press Ctrl+C to close.

## 4. Mapping (SLAM)

> A ready-made map of the apartment, `map_01`, is already included in
> `ros2_ws/src/robot_gazebo/maps/`. Steps 4–5 are only needed to build a new map;
> to try navigation directly, skip to step 6.

Open three terminals.

```bash
# Terminal 1: apartment world (RoboCup_Home)
ros2 launch robot_gazebo room_worlds.launch.py
```

```bash
# Terminal 2: slam_toolbox + RViz (wait until the robot appears in Gazebo)
ros2 launch robot_gazebo slam.launch.py
```

```bash
# Terminal 3: keyboard control (click in this terminal before pressing keys)
ros2 run robot_gazebo teleop_key_control
```

| Key | Action |
|---|---|
| `w` | forward (tap) |
| `s` | backward (tap) |
| `a` | turn left (hold) |
| `d` | turn right (hold) |

Drive slowly through every room until the map in RViz is complete.

## 5. Save the map

With SLAM still running:

```bash
cd $PART1/ros2_ws/src/robot_gazebo/maps
ros2 run nav2_map_server map_saver_cli -f "map_01" --ros-args -p map_subscribe_transient_local:=true
```

This writes `map_01.pgm` and `map_01.yaml`. Then press Ctrl+C in all terminals.

## 6. Navigation

Open two terminals.

```bash
# Terminal 1: world in navigation mode
ros2 launch robot_gazebo room_worlds.launch.py nav:=true
```

```bash
# Terminal 2: Nav2 + AMCL with the saved map (absolute path to the .yaml)
ros2 launch robot_gazebo navigation.launch.py map:=$PART1/ros2_ws/src/robot_gazebo/maps/map_01.yaml
```

In RViz:

1. **2D Pose Estimate:** click and drag at the robot's position to set its initial pose.
2. **Single goal:** select **2D Nav Goal**, click a target and drag to set the final heading.
   The robot plans a path and drives there.
3. **Multiple goals:** click the multi-point button (*Waypoint / Nav Through Poses Mode*) in the
   lower-left panel, place each target with the goal tool (click and drag), then click the button
   again to start following the points in order.


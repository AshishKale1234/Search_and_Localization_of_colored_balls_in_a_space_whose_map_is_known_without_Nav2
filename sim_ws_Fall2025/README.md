# Search and Localization of Colored Balls  
## Known Map Environment (Without Nav2) — ROS 2

This project implements **autonomous search and localization of colored spherical objects**
in a **known indoor environment** using **TurtleBot3** and **ROS 2**, **without using the Nav2 navigation stack**.

The robot navigates through predefined waypoints, performs active scanning, detects colored balls
using camera perception, validates detections using LiDAR, estimates object positions in the global
map frame, and publishes color-specific object locations.

---

## Supported ROS 2 Distributions
- **ROS 2 Humble** (Ubuntu 22.04)
- **ROS 2 Jazzy** (Ubuntu 24.04)

---

## System Requirements
- Ubuntu 22.04 (Humble) or Ubuntu 24.04 (Jazzy)
- ROS 2 Desktop installation
- Gazebo
- TurtleBot3 packages
- AMCL + Map Server (known-map localization)
- OpenCV (`cv_bridge`)

---

## Environment Setup
Set the TurtleBot3 model before running:

```bash
export TURTLEBOT3_MODEL=waffle
```

(Optional, persistent):
```bash
echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
```

---

## Build Instructions
After cloning or downloading this repository:

```bash
cd sim_ws_Fall2025
colcon build --symlink-install
source install/local_setup.bash
```

⚠️ The workspace must be sourced in every new terminal.

---

## Running Task 3 — Search and Localization

### 1) Launch the known-map navigation stack + spawn objects (Task 3)
Per the course project instructions, **Task 3 uses `navigator.launch.py`** and enables object spawning:

```bash
ros2 launch turtlebot3_gazebo navigator.launch.py spawn_objects:=true
```

This launches:
- Gazebo simulation (house world)
- Map server (loads the saved map from Task 1)
- AMCL localization
- RViz for visualization / debugging
- Ball spawning (red / green / blue)

> Note: The same `navigator.launch.py` launch file is used for **Tasks 2 and 3**.
> For Task 2, the provided command is typically:
> `ros2 launch turtlebot3_gazebo navigator.launch.py static_obstacles:=true`

---

### 2) Run the Task 3 node
Open a **new terminal**, source the workspace, and run:

```bash
cd sim_ws_Fall2025
source install/local_setup.bash
python3 src/turtlebot3_gazebo/src/lab4/task3.py
```

---

## Search Strategy
- The robot navigates through a predefined set of waypoints
- At each waypoint, it performs a **360° scanning behavior**
- Camera-based color segmentation detects candidate objects
- LiDAR data validates object range and bearing
- Object positions are transformed into the **map frame**

---

## Published Topics
Detected object positions are published as `geometry_msgs/Point` (z can be 0):

- `/red_pos`   — Red ball location
- `/green_pos` — Green ball location
- `/blue_pos`  — Blue ball location

Each topic publishes the estimated object position in the global map frame.

---

## Project Structure
Relevant files:

```
sim_ws_Fall2025/
 └── src/
     └── turtlebot3_gazebo/
         └── src/
             └── lab4/
                 └── task3.py   # Search, detection, and localization logic
```

---

## Key Features
- Autonomous search in a known environment
- Waypoint-based exploration strategy
- Camera-based color detection
- LiDAR-assisted validation
- Global-frame object localization
- Color-specific position publishing (`/red_pos`, `/green_pos`, `/blue_pos`)
- No Nav2 usage

---

## Notes
- Localization is handled using AMCL on a known map (from Task 1)
- Navigation is implemented using custom planning/control logic (no Nav2 planners/controllers)
- Ball locations are randomized during grading; hardcoding is not reliable

---

## Author
Ashish Kale  
Autonomous Systems — ROS 2

# Search and Localization of Colored Objects (No Nav2)

Video demo: https://www.dropbox.com/scl/fi/q5rjpag56otxfj9ec58rk/task3.mp4?rlkey=fvukagmbe9veauqqqq290m7ya&st=vbvikoyw&dl=0

- Autonomous search and localization of colored spherical objects in a **known indoor environment**
- Implemented using **TurtleBot3** and **ROS 2**
- Known map localization using **AMCL**
- Waypoint-based exploration with active **360° scanning**
- Camera-based color detection for object identification
- LiDAR-assisted validation for range and bearing estimation
- Object positions estimated in the **global map frame**
- Color-specific object locations published to ROS topics:
  - `/red_pos`
  - `/green_pos`
  - `/blue_pos`
- Fully custom navigation, perception, and localization pipeline
- **No use of the Nav2 navigation stack**
- Designed and evaluated in **Gazebo simulation**

Detailed build and run instructions are provided inside  
**`sim_ws_Fall2025/README.md`**

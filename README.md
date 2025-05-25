# Doodle Navigation
**Draw your route, and let the robot do the rest: intuitive robot path planning by simply doodling on a map.**

---

## Project Overview
Doodle Navigation is my first self-developed robot navigation project, created as I taught myself robotics. This system lets you plan a robot's path by drawing directly on a bird’s-eye map of the environment—no coding, no complex mapping, just intuitive touch!

- A ceiling-mounted camera captures a live top-down image of the robot's workspace.
- An Android app (built with MIT App Inventor) lets you draw the robot's start, end, and path.
- The robot receives this map, analyzes it using a sliding window + color recognition, and extracts the path.
- The robot then follows your drawn path.

---

## Features

- **Intuitive path planning:** Just doodle the route you want your robot to take.
- **No expensive sensors:** Uses only a camera and simple vision techniques.
- **Automatic path extraction:** Robot detects your path using dominant color detection and grid analysis.
- **Open-loop movement:** Robot follows the generated path, demonstrating core navigation concepts.

---

## Requirements

### Hardware

- Raspberry Pi robot (tested on a 6-wheel differential drive rover)
- Raspberry Pi Camera (mounted overhead)

### Python Libraries

- `numpy`
- `opencv-python`
- `RPi.GPIO`
- `picamera` (for capturing images from the Pi camera)

### Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/roboticistjoseph/doodle_navigation.git

# Install required Python libraries
pip install numpy opencv-python
# For Raspberry Pi camera and GPIO, install via apt or pip as needed:
sudo apt-get install python3-picamera python3-rpi.gpio
```

---

## How to Run

1. **Set up hardware:**  
   - Mount the Pi camera to cover the robot's workspace from above.
   - Power up your Raspberry Pi robot.

2. **Draw the path:**  
   - Use the MIT App Inventor app (see [`Path Navigation.pdf`](https://github.com/roboticistjoseph/doodle_navigation/blob/main/doc/path_navigation_report.pdf) for details) to capture the workspace image and draw the start (red), end (green), and path (blue).
   - Upload the annotated image to the robot via the app/webserver.

3. **Run the navigation code:**  
   - On your Raspberry Pi:
     ```bash
     python3 navigate.py
     ```
   - The robot will process the image and follow your drawn path.

---

## Limitations & Future Work

- Robot drift may occur due to lack of real-time feedback/odometry.
- Lighting and excess colors in the environment can affect detection accuracy.
- The drawn path cannot cross itself; works best for simple routes.

---

## License

MIT License (or specify your own).

---

**This project was my introduction to robotics and robot navigation—feedback and suggestions welcome!**

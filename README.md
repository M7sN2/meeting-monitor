
#  Blocked Item Detector using YOLOv8

This project uses **YOLOv8** to detect prohibited items (like phones or laptops) in real-time, especially for environments like meetings or classrooms where distractions are not allowed.

##  Features

- Uses YOLOv8 for real-time object detection
- Detects items like phones, laptops, etc.
- Alerts/logs if any denied object is seen in the frame
- Can be extended with sound alerts or logging

##  Example Denied Items

```python
denied_items = ["cell phone", "laptop", "tv", "remote"]
```
## Requirements
```
pip install ultralytics opencv-python
```
## how to run
```
python blocked-things.py
```
- Camera starts, and any detected "denied" object will trigger an alert.
- Press q to quit.

###  Resources
- YOLOv8 Official Doc
- Ultralytics GitHub

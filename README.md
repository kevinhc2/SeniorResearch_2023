# SeniorResearch_2023
Novel Rule-Based Drone Path Planning

Requires Tello Drone

/syslab_ui: folder contains mostly old programs testing different parts of project like obj detection and subject focus

/liveyolotest.py: requires you to connect to the drone before executing the program

/yolov4-tiny.cfg: config file for yolov4-tiny model

/yolov4-tiny.weights: weights for yolov4-tiny model. Trained on COCO dataset

/updated_pathplanning: old version of obj following algorithm

When liveyolotest.py starts, it will takeoff right away but may take several seconds to connect to drone camera

You can click on a bounding box once to start following the object, and click it again to cancel the spotlight

# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:13:27 2024

@author: ALI CAN
"""

from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs

path = "best_yolov8s1.pt"

model = YOLO(path) 

labels = ['Front Opening', 'pallet']

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    kare = np.asanyarray(color_frame.get_data())

    front_openings = []

    img = kare  # cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    results = model.predict(img, verbose=False, conf=0.10)

    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        score = results[0].boxes.conf[i]
        label = results[0].boxes.cls[i]
        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
        name = labels[label]

        bbox_center_x = (x1 + x2) // 2
        bbox_center_y = (y1 + y2) // 2

        derinlik = depth_image[bbox_center_y, bbox_center_x]
        derinlik_mm = derinlik * 0.01  

        if score <= 0.5:
            continue

        if name == 'Front Opening':
            front_openings.append((x1, y1, x2, y2))

        cv2.rectangle(kare, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = name + ' ' + str(format(score, '.2f'))
        #cv2.putText(kare, text, (x1, y1 - 10), font, 1.2, (255, 0, 255), 2)
        #cv2.putText(kare, f"{derinlik_mm:.2f}m", (bbox_center_x, bbox_center_y), font, 1.2, (255, 255, 0), 2)

    if len(front_openings) == 2:
        x1_1, y1_1, x2_1, y2_1 = front_openings[0]
        x1_2, y1_2, x2_2, y2_2 = front_openings[1]

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_center_x = (inter_x1 + inter_x2) // 2
            inter_center_y = (inter_y1 + inter_y2) // 2

            cv2.circle(kare, (inter_center_x, inter_center_y), 5, (0, 255, 255), -1)

            # Distance calculation
            distance = depth_frame.get_distance(inter_center_x, inter_center_y)

            #depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            #point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [inter_center_x, inter_center_y], distance)

            #print(f"3D coordinates: {point_3d}")
            
            cv2.putText(kare, f"distance is: {distance:.2f}m", (30, 30), font, 1.2, (255, 255, 255), 2)

    cv2.imshow('Camera', kare)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()

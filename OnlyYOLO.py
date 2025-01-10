# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:33:53 2024

@author: ALI CAN
"""

from ultralytics import YOLO
import cv2
#import numpy as np

path = "best_yolov8s1.pt"

model = YOLO(path) 

labels=[ 'Front Opening', 'pallet']
font = cv2.FONT_HERSHEY_SIMPLEX

kamera = cv2.VideoCapture(0)

while True:
    ret, frames = kamera.read()
    results = model.predict(frames)
    
    front_openings = []
    
    for i in range (len(results[0].boxes)):
        x1,y1,x2,y2 = results[0].boxes.xyxy[i]
        score=results[0].boxes.conf[i]
        label=results[0].boxes.cls[i]
        x1,y1,x2,y2,score,label=int(x1),int(y1),int(x2),int(y2),float(score),int(label)
        name=labels[label]
        
        bbox_center_x = (x1 + x2) // 2
        bbox_center_y = (y1 + y2) // 2


        if score <= 0.5:
          continue
      
        if name == 'Front Opening':
            front_openings.append((x1,y1,x2,y2))
            
        
        cv2.rectangle(frames, (x1,y1), (x2,y2), (255,0,0), 2)
        text= name+' '+str(format(score, '.2f'))
        cv2.putText(frames, text,(x1, y1-10), font, 1.2, (255,0,255), 2)
    
    if len(front_openings) == 2:
        x1_1, y1_1, x2_1, y2_1 = front_openings[0]
        x1_2, y1_2, x2_2, y2_2 = front_openings[1]
        
        # Compute the intersection of the two bounding boxes
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_center_x = (inter_x1 + inter_x2)//2
            inter_center_y = (inter_y1 + inter_y2) // 2
            
            cv2.circle(frames, (inter_center_x, inter_center_y), 5, (0,255,255))
        
    
    cv2.imshow('kamera', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()



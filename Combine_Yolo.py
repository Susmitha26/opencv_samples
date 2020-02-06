import numpy as np
import imutils
import time as t
import cv2
#from demog import*
#from yolo_video import*
#from no_plate_detection_video import*
#from yolo_edit import*
from fire-detection-cnn-master import *
cap = cv2.VideoCapture("zebg.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames per second : ' + str(fps))
print('Total number of frames : ' + str(frame_count))
def _main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Done")
            break
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        time = round(time / 1000, 2)
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #main(frame,frame_no,time)
        #Persons(frame,frame_no,time)
        #number_plate(frame,frame_no,time)
        #animal(frame,frame_no,time)
	res=detect_fire(frame,frame_no,time)
	print(res)
        #cv2.imshow("Detection_window", frame)
if __name__ == '__main__':
   _main()

import cv2
import numpy as np
import json
 
video_file = "video_sample1.mp4"
video = cv2.VideoCapture(video_file)
start_time=cv2.getTickCount()
#fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
 
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
 
    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    #fgmask = fgbg.apply(frame)
    no_red = cv2.countNonZero(mask)
    no=np.count_nonzero(mask)
    result={}
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(no_red)
    #if no_red>1000:
        #cv2.imshow('Output',output)
        #cv2.imshow('Output', fgmask)
    #for cnt in contours:
        #area = cv2.contourArea(cnt)
        #if int(no_red) > 1000:
            #cv2.drawContours(frame, cnt, -1, (10, 150, 150), 6)
            #cv2.imshow("output", frame)
            #cv2.waitKey()
    #print("output:", frame)
    if int(no_red) > 20000:
        cv2.putText(frame,"Fire detected",(4,3),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,150,0),3)
        end_time=cv2.getTickCount()
        time=end_time-start_time
        time1=time/1000
        result["detection"]=1
        result["frame"]=video.get(cv2.CAP_PROP_POS_FRAMES)
        result['time']=time1
    else:
        cv2.putText(frame, "Not detected", (4, 3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 150, 0), 5)
        result["detection"]=0
        result["frame"]=video.get(cv2.CAP_PROP_POS_FRAMES)
    cv2.imshow("output",frame)
    res=json.dumps(result)
    print(res)
    #print(int(no_red))
    #print("output:".format(mask))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()
video.release()

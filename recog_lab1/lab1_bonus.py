import numpy as np
import cv2
import time
def edit(gray_image):
    output = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(output, (300, 100), (500, 400), (255, 138, 229), 2)
    cv2.line(output, (50, 50), (600, 50), (223, 255, 124), 2)
    return output
# The duration in seconds of the video captured
capture_duration = 5


cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
outgray = cv2.VideoWriter('outputgray.avi',fourcc, 20.0, (640,480))
start_time = time.time()
while( int(time.time() - start_time) < capture_duration ):
    ret, frame = cap.read()
    if ret==True:
        #cv2.imshow("Video", frame)
        frame = cv2.flip(frame,1)
        out.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
cap_import = cv2.VideoCapture("output.avi")
while(cap_import.isOpened()):
    ret, frame = cap_import.read()
    if ret == True:
        # cv2.imshow("Video", frame)
        #frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outgray.write(edit(gray))
        cv2.imshow('frame', edit(gray))
        cv2.waitKey(40)
    else:
        break
cap_import.release()
outgray.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from PIL import Image

video=[]
width = 406
height = 720
for i in range(0,42):
    #a=Image.open(r'C:\Users\Anna.Kravchenko\PycharmProjects\recognotion_lab3\frames_of_interest{0}-{1}\anchor.png'.format(2*i,2*i+1))
    #t=Image.open(r'C:\Users\Anna.Kravchenko\PycharmProjects\recognotion_lab3\frames_of_interest{0}-{1}\predicted_anchor.png'.format(2 * i,2 * i + 1))
    video.append(cv2.imread(r'C:\Users\Anna.Kravchenko\PycharmProjects\recognotion_lab3\frames_of_interest{0}-{1}\error_image.png'.format(2*i,2*i+1)))
    video.append(cv2.imread(r'C:\Users\Anna.Kravchenko\PycharmProjects\recognotion_lab3\frames_of_interest{0}-{1}\error_image.png'.format(2 * i,2 * i + 1)))


out = cv2.VideoWriter('ebma_error.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, (width,height))

for i in range(len(video)):
    out.write(video[i])
out.release()
cv2.waitKey(0)



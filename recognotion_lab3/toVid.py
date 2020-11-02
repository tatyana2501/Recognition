import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
filepathPredi = r'C:\Users\Танюша\PycharmProjects\recognotion_lab3\Diamond_search\predicted_image\*'
filePath = r'C:\Users\Танюша\PycharmProjects\recognotion_lab3\anchor_frame\*'
width = 406
height = 720
img_array = []
#i = 1
files = glob.glob(filePath)
filesSorted = sorted(files, key = lambda name: int(name[len(filePath)+4:-11]))
filesPredicted = glob.glob(filepathPredi)
filesPredictedSorted = sorted(filesPredicted, key=lambda name: int(name[85:-14]))
for i in np.arange(0,len(filesPredictedSorted)-1,2):
    #print(filename)
    #img = cv2.imread(filename)
    img = Image.open(filesSorted[i+1])
    imgPredi = Image.open(filesPredictedSorted[i])
    backtorgbimgPredi = cv2.cvtColor(np.array(imgPredi, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    backtorgbimg = cv2.cvtColor(np.array(img, dtype = np.uint8), cv2.COLOR_GRAY2RGB)
    size = (width, height)
    img_array.append(backtorgbimgPredi)
    img_array.append(backtorgbimg)

out = cv2.VideoWriter('project24.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

for i in range(len(img_array)):
    #print(cv2.utils.dumpInputArray(img_array[i]))
    out.write(img_array[i])
out.release()
import cv2 as cv
import numpy as np
from PIL import Image
import time

# відносна кількість правильно суміщених ознак
# похибка локалізації (відстань між реальним розміщенням предмета в кадрі та розпізнаним)
# відносний час обробки фото в залежності від розміру зображення

def AKAZE(im1,im2):
    start = time.time()
    detector = cv.AKAZE_create()
    (kps1,descs1)=detector.detectAndCompute(im1,None)
    (kps2,descs2)=detector.detectAndCompute(im2,None)
    finish = time.time()
    bf=cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs2,k=2)
    matched1 = []
    matched2 = []
    match,mm=[],[]
    t.append(finish-start)
    nn_match_ratio = 0.8 # Nearest neighbor matching ratio
    for m, n in matches:
        mm.append((m.distance + n.distance) / 2.0)
        if m.distance < nn_match_ratio * n.distance:
            match.append(cv.DMatch(len(matched1), len(matched2), 0))
            matched1.append(kps1[m.queryIdx])
            matched2.append(kps2[m.trainIdx])
    m2=np.mean(mm)
    res = np.empty((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], 3), dtype=np.uint8)
    im3 = cv.drawMatches(im1,matched1,im2,matched2,match,res)
    good.append(len(match)/len(kps1))
    loc.append(m2)
    f.write("%9f%12f%10f\n"%(len(match)/len(kps1),m2,finish-start))
    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))
    print("Matches: {}".format(len(matched1)))
    print("Matches: {}".format(len(matched2)))
    print("{}".format(matches))
    print("{}".format(matched1))
    print("{}".format(matched2))
    # cv.imshow("AKAZE Match!!",im3)
    # cv.waitKey(0)

#size of images is too big, so i need to resize it
#and images for some reason are horizontal -_-
def resize_image(input_image_path,i):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print(original_image.size)
    resized_size = int(width/4),int(height/4)
    print(resized_size)
    resized_image = original_image.resize(resized_size)
    if i !=104:
        resized_image.save("Mr.MiniPotato/im{0}.jpg".format(i))
    else:
        original_image.save("Mr.MiniPotato/im{0}.jpg".format(i))

im, t,good,loc=[],[],[],[]
#resize_image("Mr.Potato/im0.jpg",0)
#im1 = cv.imread("Mr.MiniPotato/im0.jpg")
im1=cv.imread("turtle/turtle.jpg")

#f=open('res_MrPotato.txt','w')
f=open('res_turtle.txt','w')
f.write('Percent   Localization     Time\n')
for i in range(101):
     print(i)
     #resize_image("Mr.Potato/im{0}.jpg".format(i + 1),i+1)
     #im.append(cv.imread("Mr.MiniPotato/im{0}.jpg".format(i+1)))
     im.append(cv.imread("turtle/turtle ({0}).jpg".format(i+1)))
     AKAZE(im1,im[i])
print(good)
print(loc)
print(t)
f.close()
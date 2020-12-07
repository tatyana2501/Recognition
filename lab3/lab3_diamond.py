import itertools as it
import cv2
import numpy as np
import math
import time
import os
from PIL import Image
from skimage.io import imsave
import matplotlib.pyplot as plt

forecast=[]

def v2a(s):
    i = 0
    capture=cv2.VideoCapture(s)
    frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    gray_image = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
    ret=True
    while(i<frameCount and ret):
        ret,frame[i] = capture.read()
        if ret==True:
            gray_image[i] = cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY)
           # frame[i] = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            #cv2.imshow('frame', frame[i])
            #cv2.waitKey(30)
        else:
            break
        i += 1
    capture.release()
    return gray_image
def get_name(filename,frame_number,type):
   return filename.split('.')[0] + '_' + str(frame_number) + '_'+ type +'.jpg'

def create_folder(path,name):
    try:
        os.makedirs(path + name)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

def get_target_block(top_left, target_frame, windowSize):
    num_rows = windowSize
    num_cols = windowSize

    block = np.zeros((num_rows, num_cols), dtype=target_frame.dtype)
    # cases that target frame block outside the image
    for i,j in it.product(range(np.int(top_left[0]),np.int(top_left[0])+windowSize),range(np.int(top_left[1]),np.int(top_left[1])+windowSize)):
        try:
            block[(i-np.int(top_left[0])),(j-np.int(top_left[1]))] = target_frame[i,j]
        except:
            block[(i-np.int(top_left[0])),(j-np.int(top_left[1]))] = 0

    return block

def get_makro_block(top_left,target_frame,windowSize,searchWindowSize):
    arrangeSearchWindowSize = int((searchWindowSize-windowSize)/2)
    block = np.zeros((searchWindowSize, searchWindowSize), dtype=target_frame.dtype)
    if(top_left[0]-arrangeSearchWindowSize<0 or top_left[0]+searchWindowSize-1-arrangeSearchWindowSize>target_frame.shape[0]-1 or
            top_left[1]-arrangeSearchWindowSize < 0 or top_left[1]+searchWindowSize-1-arrangeSearchWindowSize > target_frame.shape[1] -1):
        for i,j, in it.product(range(top_left[0]-arrangeSearchWindowSize,top_left[0]+arrangeSearchWindowSize+windowSize),
                               range(top_left[1]-arrangeSearchWindowSize,top_left[1]+arrangeSearchWindowSize+windowSize)):
            try:
                block[(i-(top_left[0]-arrangeSearchWindowSize)),(j-(top_left[1]-arrangeSearchWindowSize))] = target_frame[i,j]
            except:
                block[(i - (top_left[0]-arrangeSearchWindowSize)), (j - (top_left[1]-arrangeSearchWindowSize))] = 0
        return block



    block = target_frame[top_left[0]-arrangeSearchWindowSize:top_left[0] + searchWindowSize -arrangeSearchWindowSize,
            top_left[1]-arrangeSearchWindowSize:top_left[1] + searchWindowSize -arrangeSearchWindowSize]

    # for i,j, in it.product((top_left[0]-arrangeSearchWindowSize,top_left[0]+arrangeSearchWindowSize+windowSize),
    #                        (top_left[1]-arrangeSearchWindowSize,top_left[1]+arrangeSearchWindowSize+windowSize)):
    #     block[(i-top_left[0]),(j-top_left[1])] = target_frame[i,j]
    return block



def calculate_PSNR(filename, frame, predicted_frame, anchor_frame,type):
    image_error = abs(np.array(predicted_frame, dtype=np.float16) - np.array(anchor_frame, dtype=np.float16))
    image_error_name = filename.split('.')[0] + '_' +  type + '_' + str(frame) + '_error.jpg'
    if (type == 'ES'):
        folder_name = 'Exhaustive_search'
    else:
        folder_name = 'Diamond_search'
    imsave(folder_name + '/error_image/'+image_error_name, np.array(image_error,dtype=np.uint8))

    image_error = np.array(image_error, dtype=np.uint8)
    MSE = (image_error ** 2).mean()
    print(type + '_' + str(frame) + '_' + 'MSE : {0:.2f}'.format(round(MSE,2)))
    PSNR = 10 * math.log10((255 ** 2) / MSE)
    print(type + '_' + str(frame) + '_' + 'PSNR :', "{0:.2f}".format(round(PSNR,2)))
    return MSE,PSNR
class Exhaustive_searcher():
    def __init__(self, windowSize, searchWindowSize, height, width):
        self.windowSize = windowSize
        self.height = height
        self.width = width
        self.searchWindowSize = searchWindowSize

    def predict_image(self, anchor_frame, target_frame):
        windowSize = self.windowSize
        searchWindowSize = self.searchWindowSize
        i = 0

        # creation of predicted frame
        predicted_frame = np.zeros((self.height, self.width), dtype=np.uint8)

        for (block_row, block_column) in it.product(range(0, self.height - (windowSize -1), windowSize), range(0, self.width - (windowSize- 1),
                                                                                                 windowSize)):  # height, width for anchor frame
            anchor_block = anchor_frame[block_row:block_row + windowSize, block_column:block_column + windowSize]
            # error if array size is not same with windowSize
            assert anchor_block.shape == (windowSize, windowSize)
            # initialize
            min_distance = np.inf

            for search_block_row, search_block_col in it.product(range(-searchWindowSize, searchWindowSize + windowSize),
                                                                 range(-searchWindowSize, searchWindowSize + windowSize)):
                # creation of new rectangele window windowSize + searchWindowSize
                up_left_corner = ((block_row + search_block_row), (block_column + search_block_col))
                down_right_corner = (
                (block_row + search_block_row + windowSize - 1), (block_column + search_block_col + windowSize - 1))
                print(i)
                i+=1

                # create the target window with the areas deleted outside the image
                target_block = get_target_block(up_left_corner, target_frame, windowSize)
                if not(target_block.shape == (windowSize,windowSize)):
                    continue
                # error if array size is not same with windowSize
                #print(target_block.shape)
                #assert target_block.shape == (windowSize, windowSize)

                distance = np.array(target_block, dtype=np.float16) - np.array(anchor_block, dtype=np.float16)

                norm_distance = np.linalg.norm(distance, ord=1)

                if norm_distance < min_distance:
                    min_distance = norm_distance
                    matching_block = target_block

            predicted_frame[block_row:block_row + windowSize, block_column:block_column + windowSize] = matching_block


        return predicted_frame
class NV12Read_Convert:
    def __init__(self, filename, width, height, frame):
        self.height = height
        self.width = width
        self.frame = frame
        self.filename = str(filename)

    def Read_Convert(self):
        Vstream_y = open(self.filename, 'rb')
        Vstream_uv = open(self.filename, 'rb')

        y = np.zeros((self.height, self.width))
        conv_image_filename = self.filename.split('.')[0] + '_' + str(self.frame) + '.bmp'
        conv_image = Image.new("RGB", (self.width, self.height))

        pix = conv_image.load()

        file_size = os.path.getsize(self.filename)
        # y = 1 # u,v =0.5
        frame_size = int(1.5 * self.height * self.width)
        # number of frames should be 300
        number_of_frames = file_size / frame_size
        # print(number_of_frames)

        start_frame = int(frame_size * self.frame)
        uv_start = start_frame + int(self.width * self.height)
        # print('start_frame : ', start_frame)
        # seek from zero position
        Vstream_y.seek(start_frame, 0)

        for i, j in it.product(range(0, self.height), range(0, self.width)):
            y[i, j] = (ord(Vstream_y.read(1)))
        return y

        conv_image.save(conv_image_filename)
class Diamond_searcher():
    def __init__(self,windowSize,searchWindowSize,height,width):
        self.windowSize = windowSize
        self.searchWindowSize = searchWindowSize #20
        self.height = height
        self.width = width

    def predict_image(self,anchor_frame, target_frame):
        windowSize = self.windowSize
        searchWindowSize = self.searchWindowSize

        #creation of predicted frame
        #predicted_frame = np.zeros((self.height, self.width), dtype=np.float16)
        makro_block = np.zeros((searchWindowSize,searchWindowSize), dtype=np.float16)

        for (block_row, block_column) in it.product(range(0, self.height - (windowSize - 1), windowSize),
                                                    range(0, self.width - (windowSize - 1),
                                                          windowSize)):

            anchor_block = anchor_frame[block_row:block_row + windowSize, block_column:block_column + windowSize]
            # error if array size is not same with windowSize
            assert anchor_block.shape == (windowSize, windowSize)
            top_left = block_row,block_column


            makro_block = get_makro_block(top_left,target_frame,windowSize,searchWindowSize)
            if not (makro_block.shape == (searchWindowSize, searchWindowSize)):
                continue
            start_point = [[0,0]]


            #LDSP search
            while(True):
                call,start_point = Diamond_searcher.LDSP_search(self,start_point, anchor_block, makro_block)
                if(call == 0):
                    break

            #start_point[0] = top_left[0] + start_point[0]
            #start_point[1] = top_left[1] + start_point[1]
            #SDSP search

            matching_block = Diamond_searcher.SDSP_search(self,start_point,anchor_block,makro_block)
            #predicted_frame[block_row:block_row+windowSize,block_column:block_column+windowSize] =matching_block

            # print("~~~~~~~")
            # print(block_row)
            # print(block_column)
            # print("~~~~~~~")
        #return predicted_frame
        return forecast

    def LDSP_search(self,global_start, anchor_block, makro_block):
        windowSize = self.windowSize
        searchWindowSize = self.searchWindowSize
        call = 1
        LDSP_searching_points = [[2, 0], [1, 1], [0, 2], [-1, 1], [-2, 0], [-1, -1], [0, -2], [1, -1]]
        # initialize
        min_distance = np.inf
        target_block = get_target_block(
            (global_start[0][0] + int(searchWindowSize / 2) - int(windowSize / 2),
            global_start[0][1] + int(searchWindowSize / 2) - int(windowSize / 2)),
            makro_block, windowSize)

        distance = np.array(target_block, dtype=np.float16) - np.array(anchor_block, dtype=np.float16)

        norm_distance = np.linalg.norm(distance, ord=1)
        start_point = [[0, 0]]
        if norm_distance < min_distance:
            min_distance = norm_distance
            matching_block = target_block
            center_block = target_block


        for i in range(0, 8):
            target_block = get_target_block((global_start[0][0] + (LDSP_searching_points[i])[0] + int(searchWindowSize/2) - int(windowSize/2),
                                            global_start[0][1] + (LDSP_searching_points[i])[1] + int(searchWindowSize/2) - int(windowSize/2)),makro_block, windowSize)
            if not (target_block.shape == (windowSize, windowSize)):
                continue

            distance = np.array(target_block, dtype=np.float16) - np.array(anchor_block, dtype=np.float16)

            norm_distance = np.linalg.norm(distance, ord=1)

            if norm_distance < min_distance:
                min_distance = norm_distance
                matching_block = target_block
                start_point[0] = LDSP_searching_points[i]


        if (np.array_equal(center_block,matching_block)):
            call = 0
        global_start[0][0], global_start[0][1] = global_start[0][0] + start_point[0][0], global_start[0][1] + start_point[0][1]

        return call,global_start

    def SDSP_search(self,start_point,anchor_block,makro_block):
        windowSize = self.windowSize
        searchWindowSize = self.searchWindowSize
        SDSP_searching_points = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        # initialize
        v = SDSP_searching_points[0]
        min_distance = np.inf
        target_block = get_target_block(
            (start_point[0][0] + int(searchWindowSize / 2) - int(windowSize / 2),
             start_point[0][1] + int(searchWindowSize / 2) - int(windowSize / 2)),
            makro_block, windowSize)

        distance = np.array(target_block, dtype=np.float16) - np.array(anchor_block, dtype=np.float16)

        norm_distance = np.linalg.norm(distance, ord=1)

        if norm_distance < min_distance:
            min_distance = norm_distance
            matching_block = target_block
            center_block = target_block
            v=[0,0]

        for i in range(0,4):
            target_block = get_target_block(
                (start_point[0][0] + (SDSP_searching_points[i])[0] + int(searchWindowSize / 2) - int(windowSize / 2),
                start_point[0][1] + (SDSP_searching_points[i])[1] + int(searchWindowSize / 2) - int(windowSize / 2)),
                makro_block, windowSize)

            if not (target_block.shape == (windowSize, windowSize)):
                continue

            distance = np.array(target_block, dtype=np.float16) - np.array(anchor_block, dtype=np.float16)

            norm_distance = np.linalg.norm(distance, ord=1)

            if norm_distance < min_distance:
                min_distance = norm_distance
                matching_block = target_block
                v=SDSP_searching_points[i]
        temp=[]
        temp.append(start_point[0])
        temp.append(v)
        forecast.append(temp)
        print("````````````````")
        print(start_point[0])
        print(v)
        print("````````````````")
        return matching_block

v1 = v2a("vid1.mp4")
filename = 'vid1.mp4'
width = 406
height = 720
frame = 0
total_frame_number = len(v1)-1
windowSize = 100
searchWindowSize = 200
arrangeSearchWindowSize = int((searchWindowSize-windowSize)/2)
assert (searchWindowSize>=windowSize), "Search window size cannot be smaller than window size"

path = os.getcwd()
create_folder(path, '/anchor_frame')
create_folder(path, '/Diamond_search/predicted_image')
create_folder(path, '/Diamond_search/error_image')

def build(anchor_frame, target_frame, forecast, windowSize, searchWindowSize):
    predicted_frame = np.zeros((anchor_frame.shape[0], anchor_frame.shape[1]), dtype=np.float16)
    makro_block = np.zeros((searchWindowSize, searchWindowSize), dtype=np.float16)
    i=0
    for (block_row, block_column) in it.product(range(0, predicted_frame.shape[0] - (windowSize - 1), windowSize),
                                                range(0, predicted_frame.shape[1] - (windowSize - 1),
                                                      windowSize)):
        top_left = block_row, block_column
        makro_block = get_makro_block(top_left, target_frame, windowSize, searchWindowSize)
        matching_block = get_target_block(
                (forecast[i][0][0] + forecast[i][1][0] + int(searchWindowSize / 2) - int(windowSize / 2),
                forecast[i][0][1] + forecast[i][1][1] + int(searchWindowSize / 2) - int(windowSize / 2)),
                makro_block, windowSize)
        predicted_frame[block_row:block_row + windowSize, block_column:block_column + windowSize] = matching_block
        i+=1
    return predicted_frame
psnr=[]
#Diamond Search
def callDiamondSearch(v1,filename):
    newVid =[]
    ds = Diamond_searcher(windowSize=windowSize, searchWindowSize=searchWindowSize, width=width, height=height)
    DS_total_MSE = 0
    DS_total_PSNR = 0
    target_no = 1
    anchor_no = 0
    start = time.time()

    while (anchor_no < total_frame_number):
        # target = NV12Read_Convert(filename, width, height, target_no)
        target_frame = v1[target_no]
        # anchor = NV12Read_Convert(filename, width, height, anchor_no)
        anchor_frame = v1[anchor_no]
        ds_forecast = ds.predict_image(anchor_frame=anchor_frame, target_frame=target_frame)

        ds_predicted_image = build(anchor_frame, target_frame, ds_forecast,windowSize,searchWindowSize)

        MSE, PSNR = calculate_PSNR(filename, anchor_no, ds_predicted_image, anchor_frame, 'DS')
        DS_total_MSE = DS_total_MSE + MSE
        DS_total_PSNR = DS_total_PSNR + PSNR
        psnr.append(PSNR)
        imsave(path + '/anchor_frame/' + get_name(filename, anchor_no, 'anchor'),
               np.array(anchor_frame, dtype=np.uint8))
        imsave(path + '/Diamond_search/predicted_image/' + get_name(filename, anchor_no, 'predicted'),
               np.array(ds_predicted_image, dtype=np.uint8))
        backtorgbAnchorFrame = cv2.cvtColor(np.array(anchor_frame, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        backtorgbAnchorFramePredicted = cv2.cvtColor(np.array(ds_predicted_image, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        newVid.append(backtorgbAnchorFrame)
        newVid.append(backtorgbAnchorFramePredicted)
        anchor_no = anchor_no + 2
        target_no = target_no + 2

    end = time.time()
    total_time = (end - start) / 60
    mean_time = total_time / (total_frame_number - 1)
    DS_mean_PSNR = DS_total_PSNR / (total_frame_number - 1)
    DS_mean_MSE = DS_total_MSE / (total_frame_number - 1)
    print('total number of frames processed: ', total_frame_number)
    print('window size: ', windowSize)
    print('search window size: ', searchWindowSize)
    print('total time for DS in minutes: ', '{0:.3f}'.format(round(total_time, 2)))
    print('mean time for DS in minutes: ' '{0:.3f}'.format(round(mean_time, 2)))
    print('mean PNSR for DS : ', '{0:.2f}'.format(round(DS_mean_PSNR, 2)))
    print('mean MSE for DS : ', '{0:.2f}'.format(round(DS_mean_MSE, 2)))

    return newVid

def plot_psnr(psnr):
    plt.plot(np.arange(1,44),psnr, c='magenta')
    plt.show()

newVid1 = callDiamondSearch(v1,filename)
print(psnr)
plot_psnr(psnr)
out = cv2.VideoWriter('output/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, (width,height))

for i in range(len(newVid1)):
    out.write(newVid1[i])
out.release()
cv2.waitKey(0)

print(forecast)
import cv2
import os
import matplotlib.pyplot as plt
import math
import itertools
from past.builtins import xrange
#from python_utils import show_quiver
import numpy as np


#frameWidth, frameHeight = 0, 0

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

class EBMA_searcher():
    """
    Estimates the motion between to frame images
     by running an Exhaustive search Block Matching Algorithm (EBMA).
    Minimizes the norm of the Displaced Frame Difference (DFD).
    """

    def __init__(self, N, M, R, p=1, acc=1):
        """
        :param N: Size of the blocks the image will be cut into, in pixels.
        :param R: Range around the pixel where to search, in pixels.
        :param p: Norm used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.
        :param acc: 1: Integer-Pel Accuracy (no interpolation),
                    2: Half-Integer Accuracy (Bilinear interpolation)
        """

        self.N = N
        self.M = M
        self.R = R
        self.p = p
        self.acc = acc

    def run(self, anchor_frame, target_frame):
        """
        Run!
        :param anchor_frame: Image that will be predicted.
        :param target_frame: Image that will be used to predict the target frame.
        :return: A tuple consisting of the predicted image and the motion field.
        """

        acc = self.acc
        height = anchor_frame.shape[0]
        width = anchor_frame.shape[1]
        N = self.N
        R = self.R
        p = self.p
        M = self.M

        # interpolate original images if half-pel accuracy is selected
        if acc == 1:
            pass
        elif acc == 2:
            target_frame = cv2.resize(target_frame, dsize=(width * 2, height * 2))
        else:
            raise ValueError('pixel accuracy should be 1 or 2. Got %s instead.' % acc)

        # predicted frame. anchor_frame is predicted from target_frame
        predicted_frame = np.empty((height, width), dtype=np.uint8)

        # motion field consisting in the displacement of each block in vertical and horizontal
        motion_field = np.empty((int(height / N), int(width / M), 2))
        Start_pos = np.empty((int(height / N), int(width / M), 2))

        # loop through every NxN block in the target image
        for (blk_row, blk_col) in itertools.product(xrange(0, height - (N - 1), N),
                                                    xrange(0, width - (M - 1), M)):

            # block whose match will be searched in the anchor frame
            blk = anchor_frame[blk_row:blk_row + N, blk_col:blk_col + M]
            print(blk.shape)
            print(anchor_frm.shape)

            # minimum norm of the DFD norm found so far
            dfd_n_min = np.infty

            # search which block in a surrounding RxR region minimizes the norm of the DFD. Blocks overlap.
            for (r_col, r_row) in itertools.product(range(-R, (R + N)),
                                                    range(-R, (R + M))):
                # candidate block upper left vertex and lower right vertex position as (row, col)
                up_l_candidate_blk = ((blk_row + r_row) * acc, (blk_col + r_col) * acc)
                low_r_candidate_blk = ((blk_row + r_row + N - 1) * acc, (blk_col + r_col + M - 1) * acc)
                print(up_l_candidate_blk)
                print(low_r_candidate_blk)

                # don't search outside the anchor frame. This lowers the computational cost
                if up_l_candidate_blk[0] < 0 or up_l_candidate_blk[1] < 0 or \
                                low_r_candidate_blk[0] > height * acc - 1 or low_r_candidate_blk[1] > width * acc - 1:
                    continue
                print("bulk")
                print(target_frame[list(up_l_candidate_blk)[0]:list(low_r_candidate_blk)[0]
                                , list(up_l_candidate_blk)[1]:list(low_r_candidate_blk)[1]])
                # the candidate block may fall outside the anchor frame
                candidate_blk = target_frame[list(up_l_candidate_blk)[0]:list(low_r_candidate_blk)[0]+1
                                , list(up_l_candidate_blk)[1]:list(low_r_candidate_blk)[1]+1][::acc,::acc]
                print("shape")
                print(candidate_blk.shape)
                #print(target_frame[up_l_candidate_blk, low_r_candidate_blk])
                assert candidate_blk.shape == (N,M)

                dfd = np.array(candidate_blk, dtype=np.float16) - np.array(blk, dtype=np.float16)

                candidate_dfd_norm = np.linalg.norm(dfd, ord=p)

                # a better matching block has been found. Save it and its displacement
                if candidate_dfd_norm < dfd_n_min:
                    dfd_n_min = candidate_dfd_norm
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row

            # construct the predicted image with the block that matches this block
            predicted_frame[blk_row:blk_row + N, blk_col:blk_col + M] = matching_blk

            print(str((blk_row / N, blk_col / M)) + '--- Displacement: ' + str((dx, dy)))
            Start_pos[blk_row // N, blk_col // M, 1] = N/2+N*(blk_row//N)
            Start_pos[blk_row // N, blk_col // M, 0] = N/2+N*(blk_col//N)

            # displacement of this block in each direction
            motion_field[blk_row // N, blk_col // M, 1] = dx
            motion_field[blk_row // N, blk_col // M, 0] = dy


        return predicted_frame, motion_field, Start_pos
v1=v2a("vid1.mp4")
v2=v2a("vid2.mp4")
height, width = v1[0].shape[:2]
target_frm = np.array(cv2.resize(v1[0], (width,height), interpolation=cv2.INTER_CUBIC))
anchor_frm = np.array(cv2.resize(v1[1], (width,height), interpolation=cv2.INTER_CUBIC))
#target_frm = np.reshape(target_frm[:width*height], (width, height))
#anchor_frm = np.reshape(anchor_frm[:width*height], (width, height))
os.system('mkdir -p frames_of_interest')
plt.imsave('frames_of_interest/target.png', target_frm)
plt.imsave('frames_of_interest/anchor.png', anchor_frm)

ebma = EBMA_searcher(N= 54,
                     M = 54,
                     R=100,
                     p=1,
                     acc=2)

predicted_frm, motion_field, Start_pos = \
    ebma.run(anchor_frame=anchor_frm,
             target_frame=target_frm)

# store predicted frame
plt.imsave('frames_of_interest/predicted_anchor.png', predicted_frm)

motion_field_x = motion_field[:, :, 0]
motion_field_y = motion_field[:, :, 1]
Start_Pos_x = Start_pos[:, :, 0]
Start_Pos_y = Start_pos[:, :, 1]

print(motion_field_x)
print(motion_field_y[::-1])
# show motion field
plt.imshow(v1[0])
plt.quiver(Start_Pos_x,Start_Pos_y,motion_field_x , motion_field_y[::-1] , color='magenta')
plt.show()

# store error image
error_image = abs(np.array(predicted_frm, dtype=float) - np.array(anchor_frm, dtype=float))
error_image = np.array(error_image, dtype=np.uint8)
plt.imsave('frames_of_interest/error_image.png', error_image)

# Peak Signal-to-Noise Ratio of the predicted image
mse = (np.array(error_image, dtype=float) ** 2).mean()
psnr = 10 * math.log10((255 ** 2) / mse)
print('PSNR: %s dB' % psnr)




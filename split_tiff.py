import os
import numpy as np

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import cv2

def read_Color_Image(path):
    retVal = cv2.imread(path,cv2.IMREAD_COLOR)
    if retVal is None: raise Exception("Reading Color image, something went wrong with the file name "+str(path))
    return retVal[:,:,::-1]

im = read_Color_Image('./Data/Koiwainoujo220616NEW.tif')

h,w,c =im.shape

wnew = w//4
hnew = h//4

for i in range(0,h,hnew):
    for j in range(0,w,wnew):
        cutIm = im[i:i+hnew, j:j+wnew,:]
        cv2.imwrite(os.path.join('./Data/testImage/KoiwaiSummer','koiwaisummer' + f"_{i}_{j}.png"),cutIm)
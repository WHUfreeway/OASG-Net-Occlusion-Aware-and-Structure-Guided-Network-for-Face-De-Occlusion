'''
Author:lby
function:resize the pic to size*size

'''

import argparse
import cv2
from PIL import Image
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path the photo')
parser.add_argument('--output_path', type=str, help='path the out photo')
parser.add_argument('--size', type=int, help='the photo will resize to size*size')
args=parser.parse_args()

#PIL读取图像
im=Image.open('mask.png')
#width,height=args.size,args.size
im=np.array(im)
print(im.shape)
for i in range(0,256):
    for j in range(0,256):
        if i>78 and i<=178 and j>78 and j<=178:
            im[i][j]=255
        else:
            im[i][j]=0
#im1=im.resize((width,height))
im1=Image.fromarray(im)
im1.save('center.png')
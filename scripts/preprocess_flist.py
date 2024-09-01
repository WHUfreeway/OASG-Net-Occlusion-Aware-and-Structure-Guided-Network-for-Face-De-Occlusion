#activate base
#python ./scripts/preprocess_landmark2.py
import os
from skimage import io
import face_alignment
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, help='path to the output folder')
args = parser.parse_args()

#input_path = args.path
#output_path = args.output
#if not os.path.exists(output_path):
#    os.mkdir(output_path)

str1 = []
file1 = open("/data1/fyw/datasets/celeba/celeba_val_images.flist","r",encoding="utf-8")
for line in file1.readlines():    #读取第一个文件
    str1.append(line.replace("\n",""))

images = []
for i in str1:
    str=i.replace('png','txt')
    str=str.replace('celeba_val_images','celeba_val_landmark')
    images.append(str)


images = sorted(images)
np.savetxt(args.output, images, fmt='%s')




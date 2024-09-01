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
file1 = open("D:/project/pycharmProject/pythonProject/lafin/datasets/celebA/celebA_test_masked.flist","r",encoding="utf-8")
for line in file1.readlines():    #读取第一个文件
    str1.append(line.replace("\n",""))

images = []
for i in str1:
    #str=i.replace('png','txt')
    str=i.replace('test_masked','test')
    images.append(str)


images = sorted(images)
np.savetxt(args.output, images, fmt='%s')




import cv2
import shutil
import os
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--newpath', type=str, help='path to the dataset')
args = parser.parse_args()

ext = {'.jpg','.png'}
str1=[]
file1 = open("./datasets/celebA/celebA_test.flist","r",encoding="utf-8")
for line in file1.readlines():    #读取第一个文件
    str1.append(line.replace("\n",""))

file_dir='D:/resultData/RFR_rectangle_45%'
for root, dirs, files in os.walk(file_dir):
    for file in files:
        str=int(re.sub("\D","",file))
        ori=os.path.join(root,file)
        file2=str1[str-1]
        file2=file2.replace("dataset/celebA/test","resultData/RFR_rectangle_45%")
        os.rename(ori,file2)
        #print(ori)
        #print(file2)
        #print(os.path.join(root,file))
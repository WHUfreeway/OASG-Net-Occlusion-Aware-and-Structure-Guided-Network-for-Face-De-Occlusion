import cv2
import shutil
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--newpath', type=str, help='path to the dataset')
args = parser.parse_args()

ext = {'.jpg','.png'}
str1=[]
file1 = open("./datasets/celebA/celebA_test.flist","r",encoding="utf-8")
for line in file1.readlines():    #读取第一个文件
    str1.append(line.replace("\n",""))

for str in str1:
    filename = str
    print(filename)
    target=str.replace('dataset/celebA/test','resultData/input_0%')
    shutil.copy(filename,target)
import cv2
import shutil
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--newpath', type=str, help='path to the dataset')
args = parser.parse_args()

ext = {'.jpg','.png'}

count=1
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            filename = root + '/' + file
            targetpath=args.newpath+'/'+str(count)+os.path.splitext(file)[1]
            count+=1
            shutil.copy(filename,targetpath)
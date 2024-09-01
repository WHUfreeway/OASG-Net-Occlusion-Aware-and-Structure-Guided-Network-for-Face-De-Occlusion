import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
args = parser.parse_args()

ext = {'.jpg'}


for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            portion=os.path.splitext(file)
            if portion[1]=='.jpg':
                filename=root+'/'+file
                newname=root+'/'+portion[0]+'.png'
                newname=newname.replace("\\",'/')
                filename=filename.replace("\\",'/')
                os.rename(filename,newname)


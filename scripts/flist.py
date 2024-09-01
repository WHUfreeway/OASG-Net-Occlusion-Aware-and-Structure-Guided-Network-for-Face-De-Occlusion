import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--output', type=str, help='path to the file list')
args = parser.parse_args()

ext = {'.jpg', '.png','.txt'}
#ext = {'.txt'}
images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            newpath=os.path.join(root,file)
            newpath=newpath.replace("\\",'/')
            images.append(os.path.join(newpath))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')
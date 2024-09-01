import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
args = parser.parse_args()

ext = {'.txt'}


for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            newpath = os.path.join(root,file)
            newpath = newpath.replace("\\", "/")
            f=open(newpath)
            if len(f.read())==0:
                f.close()
                print("delete:"+newpath)
                os.remove(newpath)
            f.close()


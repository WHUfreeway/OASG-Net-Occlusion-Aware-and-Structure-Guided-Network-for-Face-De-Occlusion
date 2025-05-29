import os
import argparse
from skimage import io
from scipy.misc import imresize

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the celeba img_align_celeba folder')
parser.add_argument('--output', type=str, help='path to the output folder')
args = parser.parse_args()

output_path = args.output
dataset_path = args.path

if not os.path.exists(output_path):
    os.mkdir(output_path)

test_path = os.path.join(output_path,'whn_test_images')

if not os.path.exists(test_path):
    os.mkdir(test_path)

filenames = os.listdir(dataset_path)

for filename in filenames:
    filenames = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    print(filename)
    save_path = test_path

    img = io.imread(os.path.join(dataset_path,filename))

    io.imsave(os.path.join(save_path,filename+'.png'),img)

    print(filename)






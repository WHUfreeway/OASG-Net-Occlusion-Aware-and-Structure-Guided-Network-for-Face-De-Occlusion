import os
from skimage import io
import face_alignment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the celeba img_align_celeba folder')
parser.add_argument('--output', type=str, help='path to the output folder')
args = parser.parse_args()

input_path = args.path
output_path = args.output

if not os.path.exists(output_path):
    os.mkdir(output_path)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,face_detector='sfd')


for root, dirname,filenames in os.walk(input_path):
    print("root:"+root)
    new_root=root.replace('celeba','celeba_landmark')
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    for filename in filenames:
        if filename[-3:] != 'png' and filename[-3:] != 'jpg':
            continue
        newpath=os.path.join(new_root,filename[:-4]+'.txt')
        with open(newpath, 'w') as f:
            img = io.imread(os.path.join(root,filename))
            print(newpath+'\n')
            l_pos = fa.get_landmarks(img)
            if l_pos == None:
              print(filename)
              continue
            if len(l_pos[0]) != 68:
              print(filename)
              print(len(l_pos[0]))
              continue
            for i in range(68):
                f.write(str(l_pos[0][i,0])+' '+str(l_pos[0][i,1])+' ')
            f.write('\n')




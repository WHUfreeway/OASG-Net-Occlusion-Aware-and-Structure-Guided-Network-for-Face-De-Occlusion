#activate base
#python ./scripts/preprocess_landmark2.py
import os
from skimage import io
import face_alignment
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--path', type=str, help='path to the celeba img_align_celeba folder')
#parser.add_argument('--output', type=str, help='path to the output folder')
#args = parser.parse_args()

#input_path = args.path
#output_path = args.output
#if not os.path.exists(output_path):
#    os.mkdir(output_path)

str1 = []
file1 = open("/data1/fyw/OASG-NET/fin/datasets/whn/img_align.flist","r",encoding="utf-8")
for line in file1.readlines():    #读取第一个文件
    str1.append(line.replace("\n",""))

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,face_detector='sfd')

for i in range(len(str1)):
    try:
        if i%100==0:
            print('process {}'.format(i))
        str2=str1[i]
        dirname=os.path.dirname(str2)
        dirname=dirname.replace('img','landmark')
        print(dirname)
        if not os.path.exists(dirname):
            print('process: '+dirname)
            os.mkdir(dirname)

        output_path=str2.replace('img','landmark')
        output_path=output_path.replace('png','txt')
        with open(output_path, 'w') as f:
            print(str2)
            img = io.imread(str2)
            l_pos = fa.get_landmarks(img)
            print('2')
            for i in range(68):
                f.write(str(l_pos[0][i,0])+' '+str(l_pos[0][i,1])+' ')
            f.write('\n')
            f.close()
    except:
        print("错误"+str(i))





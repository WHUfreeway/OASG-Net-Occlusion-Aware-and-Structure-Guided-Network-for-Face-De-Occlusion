import shutil
import os
import numpy as np
str1 = []
file1 = open("datasets/landmark256.flist","r",encoding="utf-8")
for line in file1.readlines():    #读取第一个文件
    line=line.replace("\n","")
    line=line.replace("rec2img_256_landmark", "rec2img_256")
    line=line.replace("txt","jpg")
    str1.append(line)

images = sorted(str1)
np.savetxt("./datasets/rec2img256.flist", images, fmt='%s')


import cv2
import os
str1 = []
file1 = open("masked_whn.flist","r",encoding="utf-8")
for line in file1.readlines():    #读取第一个文件
    str1.append(line.replace("\n",""))
for i in range(len(str1)):
     str2 = str1[i]
     img=cv2.imread(str2)
     img2 = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
     save_path = "D:/dataset/masked_whn_256/"+str(i)+".png"
     cv2.imwrite(save_path,img2)
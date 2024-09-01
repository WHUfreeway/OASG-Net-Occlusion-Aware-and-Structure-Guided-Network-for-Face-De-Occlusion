import colorsys
import copy
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from MAM import Unet as unet


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
#--------------------------------------------#
class Unet(object):
    _defaults = {
        "model_path"        : '/data1/fyw/OASG-NET/fin/CelebA_net6/Epoch10-Total_Loss0.0069-Val_Loss0.0053.pth',
        "model_image_size"  : (256, 256, 3),
        "num_classes"       : 2,
        "cuda"              : True,
        #--------------------------------#
        #   blend参数用于控制是否
        #   让识别结果和原图混合
        #--------------------------------#
        "blend"             : True
    }

    #---------------------------------------------------#
    #   初始化UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        self.net = unet(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()

        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('{} model loaded.'.format(self.model_path))

        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                        for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self ,image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image,nw,nh
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #---------------------------------------------------------#
        image = image.convert('RGB')
        
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        #---------------------------------------------------#
        #   进行不失真的resize，添加灰条，进行图像归一化
        #---------------------------------------------------#
        image, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        images = [np.array(image)/255]
        images = np.transpose(images,(0,3,1,2))

        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor)
            if self.cuda:
                images =images.cuda()

            pr, _, _ = self.net(images)[0]
            print(F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1).shape)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        #------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')

        #------------------------------------------------#
        #   将新图片转换成Image的形式
        #------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        image.save('result.png')
        #------------------------------------------------#
        #   将新图片和原图片混合
        #------------------------------------------------#
        if self.blend:
            image = Image.blend(old_img,image,0.7)
        return image

    def detect_image2(self, images):
        #print(images.shape)
        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        with torch.no_grad():
            if self.cuda:
                images = images.cuda()

            pr,mu1,mu2 = self.net(images)

            #print(F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1).shape)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            #print(pr.shape)
            pr = F.softmax(pr.permute(0,2,3,1),dim = -1).argmax(axis=-1)
            #print(pr.shape)
            #pr=pr.permute(0,3,1,2)
            #pd = pr[0]
            #seg_img = np.zeros((np.shape(pd)[0], np.shape(pd)[1], 3))
            #for c in range(self.num_classes):
            #    seg_img[:, :, 0] += ((pd[:, :] == c) * (self.colors[c][0])).astype('uint8')
            #    seg_img[:, :, 1] += ((pd[:, :] == c) * (self.colors[c][1])).astype('uint8')
            #    seg_img[:, :, 2] += ((pd[:, :] == c) * (self.colors[c][2])).astype('uint8')
            #image = Image.fromarray(np.uint8(seg_img))
            #image.show()
            #os.system("pause")

            pr = pr.view((pr.shape[0], 1, pr.shape[1], pr.shape[2]))


        # ------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        # ------------------------------------------------#
        #seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        #for c in range(self.num_classes):
        #    seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
        #    seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
        #    seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        #image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
        #image.save('result.png')
        # ------------------------------------------------#
        #   将新图片和原图片混合
        # ------------------------------------------------#
        #if self.blend:
        #    image = Image.blend(old_img, image, 0.7)
        return pr.float(), mu1, mu2

    def get_FPS(self, image, test_interval):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        images = [np.array(image)/255]
        images = np.transpose(images,(0,3,1,2))

        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor)
            if self.cuda:
                images =images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

import torchvision.transforms as transforms

if __name__ == '__main__':
    mask_model=Unet()
    input=torch.rand((4,3,256,256))
    image = Image.open("1.png").convert("RGB")
    
    transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    mask, mu1, mu2 = mask_model.detect_image2(input_tensor)

    # 4. 将预测的输出转换回图像格式
    mask = mask.squeeze().cpu().numpy()  # 移除batch维度并转换为numpy数组
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    # 5. 保存输出图像
    mask_img.save("1output.png")
    print(mask.shape)
    print(mu1.shape)
    print(mu2.shape)

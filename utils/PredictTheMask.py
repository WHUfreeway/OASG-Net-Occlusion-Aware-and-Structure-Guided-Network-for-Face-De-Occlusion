'''
Author:lby
function:Predict where the mask is.

'''
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path the photo')
parser.add_argument('--output', type=str, help='path of the out mask')
args=parser.parse_args()

# Import
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Normalize
import torch.nn.functional as F
from models import GCN
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Make sure your network has been trained with this architectural parameters
target_size = 256
num_classes = 7
num_levels = 3

# I/O
checkpoint = "./utils/gcn-levels_3-classes_0123456-valset_2-lr_0.001-lrdecay_0.2-lrmilestones_35_90_180-wdecay_0.0005-momentum_0.99-dataaugment-SunglassesHandsMasks_0.5/gcn-epoch_0100.pth"

# Colorize your labeled classes
label_colors = [
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (255, 255, 255)]


# Model loading (Resnet may take a while to download)
model = torch.nn.DataParallel(GCN(num_classes, num_levels))
model.load_state_dict(torch.load(checkpoint))
model.cuda()
model.eval()

def predict(image):
    with torch.no_grad():
        print(type(image))
        img = ToTensor()(image)
        print(img.shape)
        img = Normalize([.485, .456, .406], [.229, .224, .225])(img)
        img = Variable(img).cuda().unsqueeze(0)


        scores = model(img)  # first image in batch
        label_probs = F.log_softmax(scores[0], dim=0).cpu().detach().numpy()

        # Composite
        rgb = np.zeros((target_size, target_size, 3))
        labels = np.argmax(label_probs, axis=0)


        for l in range(len(label_probs)):
            indexes = labels == l
            for c in range(3):
                rgb[:, :, c][indexes] = label_colors[l][c]

        result = Image.fromarray(rgb.astype('uint8'))


        # Show the image which was just taken.
        # display(Image(filename))
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title(f'Original Image')
        plt.imshow(np.array(image))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title(f'Prediction Image')
        plt.imshow(result)
        plt.axis('off')
        plt.show()
        plt.close(fig)
        return result

def predict2(image):
    with torch.no_grad():
        img = ToTensor()(image)
        img = Normalize([.485, .456, .406], [.229, .224, .225])(img)
        img = Variable(img).cuda().unsqueeze(0)

        scores = model(img)  # first image in batch
        label_probs = F.log_softmax(scores[0], dim=0).cpu().detach().numpy()
        # Composite
        mask = np.zeros((target_size, target_size))
        rgb = np.zeros((target_size, target_size, 3))
        labels = np.argmax(label_probs, axis=0)

        for i in range(0, target_size):
            for j in range(0,target_size):
                if labels[i][j]==6:
                    mask[i][j]=1
        cv2.imwrite("result1.png", mask)
        return mask

image = Image.open(args.path).convert("RGB")
image = image.resize((target_size, target_size), Image.BILINEAR)
#print(image.shape)
mask=predict(image)
mask_array = np.array(mask)
print("mask_array.shape")
print(mask_array.shape)
cv2.imwrite(args.output, np.array(mask))

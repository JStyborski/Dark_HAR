from JS_Transforms import CustomScale, CustomGammaTransform

import numpy as np

import torch
import torchvision.transforms as IT
from torchvision.io import read_video, write_video, write_jpeg

filePath = r'S:\Datasets\Dark_HAR\train\Walk\Walk_10_1.mp4'
gamma = 1.0

# Read the video as a PyTorch tensor with TCHW format
videoTens = read_video(filePath, output_format='TCHW', pts_unit='sec')[0]

# Apply gamma transform to video
outVideoTransform = IT.Compose([
    CustomScale(scaleVal=1./255.),
    CustomGammaTransform(gamma=gamma),
    CustomScale(scaleVal=255.)
])
outVideoTens = outVideoTransform(videoTens)

# Swap axes to get right output format THWC and output the video
#outVideoTens = torch.transpose(outVideoTens, 1, 2)
#outVideoTens = torch.transpose(outVideoTens, 2, 3)
#write_video('outvideo_gm{}.mp4'.format(gamma), outVideoTens.to(torch.uint8), fps=30)

# Write image
#write_jpeg(outVideoTens[-1, :, :, :], 'outvideo_frame_gm{}.jpg'.format(gamma))

# Image histogram
finalGrayImgTens = torch.mean(outVideoTens[-1, :, :, :], dim=0).to(torch.uint8).detach().numpy()
pixelCountList = []
for pixelValue in range(0, 256):
    pixelCountList.append(np.sum(finalGrayImgTens == pixelValue))
print(pixelCountList)

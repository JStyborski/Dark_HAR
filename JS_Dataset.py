from JS_Sampler import SequenceSampler
from JS_Transforms import CustomScale, CustomGammaTransform

import os
import colorsys

import torch
import torchvision.transforms as IT
from torchvision.io import read_video, write_video
from torch.utils.data import Dataset


def compose_transforms(stage, gamma):
    if stage == 'train':
        return IT.Compose([
            IT.Resize(size=(112, 112)),
            IT.RandomHorizontalFlip(),
            CustomScale(scaleVal=1./255.),
            CustomGammaTransform(gamma=gamma),
            IT.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5)),
            IT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return IT.Compose([
            IT.Resize(size=(112, 112)),
            CustomScale(scaleVal=1./255.),
            CustomGammaTransform(gamma=gamma),
            IT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


# Write a video output. Assumes scaling has been done and scales it back up. Also swaps axes to get right output format
def write_output_video(lineParts2, videoTens):
    outVideoTransform = IT.Compose([CustomScale(scaleVal=255.0)])
    outVideoTens = outVideoTransform(videoTens)
    outVideoTens = torch.transpose(outVideoTens, 1, 2)
    outVideoTens = torch.transpose(outVideoTens, 2, 3)
    write_video('outvideo' + lineParts2[2].split('/')[1], outVideoTens, fps=30)

class DarkHARDataset(Dataset):
    def __init__(self, stage, dataRoot, nClasses, nFrames, gamma):
        super(DarkHARDataset, self).__init__()
        self.stage = stage
        self.dataRoot = dataRoot
        with open(os.path.join(self.dataRoot, self.stage + '.txt'), 'r') as f:
            self.lines = f.readlines()
        self.nClasses = nClasses
        self.SeqSamp = SequenceSampler(nFrames)
        self.videoTransforms = compose_transforms(self.stage, gamma)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):

        # Get the filename according to the text file at the specified index
        lineParts = self.lines[index][:-1].split('\t')

        # Read the video as a PyTorch tensor with TCHW format
        videoTens = read_video(os.path.join(self.dataRoot, self.stage, lineParts[2]), output_format='TCHW', pts_unit='sec')[0]

        # Determine the subset of frames to keep, then apply it
        keepFrames = self.SeqSamp.random(videoTens.size()[0], 'uniform')
        videoTens = videoTens[keepFrames, :, :, :]

        # Transform the video according to the given transforms composition
        videoTens = self.videoTransforms(videoTens)

        # Write an output video to view the effect of transforms and stuff
        # if lineParts[2] == 'Walk/Walk_10_1.mp4':
        if False:
            write_output_video(lineParts, videoTens)

        # Swap to CTHW output as expected by 3D CNN
        # I originally had this right after loading video but some transforms (Normalize, ColorJitter) expect TCHW
        videoTens = torch.transpose(videoTens, 0, 1)

        # Construct an empty torch tensor to hold the labels, then put a 1 at the index for ground truth
        labelTens = torch.Tensor(self.nClasses).fill_(0)
        labelTens[int(lineParts[1])] = 1

        return videoTens, labelTens

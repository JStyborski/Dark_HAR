from JS_Sampler import SequenceSampler
from JS_Transforms import CustomScale, CustomGammaTransform

import os
import colorsys

import torch
import torchvision.transforms as IT
from torchvision.io import read_video
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


class DarkHARDataset(Dataset):
    def __init__(self, stage, dataRoot, nClasses, nFrames, sampleMode, gamma):
        super(DarkHARDataset, self).__init__()
        self.stage = stage
        self.dataRoot = dataRoot
        with open(os.path.join(self.dataRoot, self.stage + '.txt'), 'r') as f:
            self.lines = f.readlines()
        self.nClasses = nClasses
        self.SeqSamp = SequenceSampler(nFrames, sampleMode)
        self.videoTransforms = compose_transforms(self.stage, gamma)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):

        # Get the filename according to the text file at the specified index
        lineParts = self.lines[index][:-1].split('\t')

        # Read the video as a PyTorch tensor with TCHW format
        videoTens = read_video(os.path.join(self.dataRoot, self.stage, lineParts[2]), output_format='TCHW', pts_unit='sec')[0]

        # Determine the subset of frames to keep, then apply it
        keepFrames = self.SeqSamp.sample(videoTens.size()[0])
        videoTens = videoTens[keepFrames, :, :, :]

        # Transform the video according to the given transforms composition
        # Some transforms (Normalize, ColorJitter) expect TCHW format, so keep this before the transpose
        videoTens = self.videoTransforms(videoTens)

        # Swap to CTHW output as expected by 3D CNN
        videoTens = torch.transpose(videoTens, 0, 1)

        # Construct an empty torch tensor to hold the labels, then put a 1 at the index for ground truth
        labelTens = torch.Tensor(self.nClasses).fill_(0)
        labelTens[int(lineParts[1])] = 1

        return videoTens, labelTens

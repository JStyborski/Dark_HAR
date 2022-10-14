import torch
import torchvision.transforms as IT


# Divide every element of the tensor by the scaleVal
class CustomScale:
    def __init__(self, scaleVal):
        self.scaleVal = scaleVal

    def __call__(self, videoTens):
        return torch.mul(videoTens, self.scaleVal)


# Raise every element of the tensor by the gamma value
class CustomGammaTransform:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, videoTens):
        return torch.pow(videoTens, self.gamma)
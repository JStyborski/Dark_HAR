import numpy as np


class SequenceSampler:
    def __init__(self, nFrames, mode):
        self.nFrames = nFrames
        self.mode = mode

    def sample(self, seqLen):
        # Return a random uniform sampling of frames
        if self.mode == 'RU':
            return np.sort(np.random.choice(seqLen, self.nFrames, replace=False))

        # Generate a Poisson sampling from the sequence length to determine which frames to keep
        elif self.mode == 'RP':
            # Set any frames that are double sampled to just one, then count the total frames sampled
            pOut = np.random.poisson(self.nFrames / seqLen, seqLen)
            pOut[pOut > 1] = 1
            pOutSum = sum(pOut)

            # If Poisson sampling is exactly the length wanted, output the indices
            # If too many samples generated, randomly pick the ones to keep
            # If too few samples generated, randomly pick extra frames to add
            if pOutSum == self.nFrames:
                return pOut.nonzero()[0]
            elif pOutSum > self.nFrames:
                return np.sort(np.random.choice(pOut.nonzero()[0], self.nFrames, replace=False))
            elif pOutSum < self.nFrames:
                zeroSampler = np.random.choice(np.where(pOut == 0)[0], self.nFrames-pOutSum, replace=False)
                return np.sort(np.concatenate((pOut.nonzero()[0], zeroSampler)))

        # Sample using the full range of frames, such that the beginning and end frames are included
        if self.mode == 'SF':
            return np.round(np.linspace(0, seqLen - 1, self.nFrames)).astype(int)

        # Sample from the first frame at a constant interval
        elif self.mode == 'SI':
            return np.arange(0, seqLen, np.floor((seqLen - 1) / (self.nFrames - 1)))[0:self.nFrames].astype(int)






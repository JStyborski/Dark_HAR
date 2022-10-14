import numpy as np

for seqLen in range(40, 51):
    for nSamples in range(9, 12):
        pOut = np.random.poisson(nSamples / seqLen, seqLen)
        pOut[pOut > 1] = 1
        pOutSum = sum(pOut)

        # If Poisson sampling is exactly the length wanted, output the indices
        # If too many samples generated, randomly pick the ones to keep
        # If too few samples generated, randomly pick extra frames to add
        if pOutSum == nSamples:
            zoop = pOut.nonzero()[0]
        elif pOutSum > nSamples:
            zoop = np.sort(np.random.choice(pOut.nonzero()[0], nSamples, replace=False))
        elif pOutSum < nSamples:
            zeroSampler = np.random.choice(np.where(pOut == 0)[0], nSamples - pOutSum, replace=False)
            zoop = np.sort(np.concatenate((pOut.nonzero()[0], zeroSampler)))

        print(len(zoop) == nSamples, zoop[-1] != nSamples)
        #print(zoop)
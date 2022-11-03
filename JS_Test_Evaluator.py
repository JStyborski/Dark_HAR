from JS_Dataset import DarkHARDataset
from JS_Evaluator import DarkHAREvaluator

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

###############
# User Inputs #
###############

# Dataset inputs
dataRoot = r'S:\Datasets\Dark_HAR'
nClasses = 10
nFrames = 20
sampleMode = 'SF' # RU: Random Uniform, RP: Random Poisson, SF: Sequential Full, SI: Sequential Interval
gamma = 0.2

# Model inputs
checkpointFile = 'checkpoints/3_nf20_gm0.2_cjAfter_smSF_atTrue_bs8_lr0.001_mm0.9_wd0.005_ep100_last.ckpt'

# Training inputs
batchSize = 8

###################################
# Define Datasets and DataLoaders #
###################################

testDataset = DarkHARDataset(stage='test', dataRoot=dataRoot, nClasses=nClasses, nFrames=nFrames, sampleMode=sampleMode, gamma=gamma)
testDataLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, num_workers=0)
print('Data Loaded', flush=True)

################
# Define Model #
################

# Load model with pretrained weights
model = r3d_18(weights=R3D_18_Weights.DEFAULT)

# Replace final classification layer
model.fc = torch.nn.Linear(512, 10)
model.cuda()

model.load_state_dict(torch.load(checkpointFile))

print('Model Loaded', flush=True)

#############################
# Training/Evaluation Setup #
#############################

# Loss function
lossFcn = torch.nn.CrossEntropyLoss()

# Instantiate evaluator from custom class
evaluator = DarkHAREvaluator(model, valDataLoader=testDataLoader, lossFcn=lossFcn)
print('Evaluator Loaded', flush=True)

############################
# Training/Evaluation Loop #
############################

print('Running Test')

evalLoss, evalAcc = evaluator.eval_epoch()

print('Test_Loss: {} | Test_Acc: {}'.format(evalLoss, evalAcc * 100))

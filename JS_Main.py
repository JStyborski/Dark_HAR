from JS_Dataset import DarkHARDataset
from JS_Trainer import DarkHARTrainer
from JS_Evaluator import DarkHAREvaluator

import time
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
nFrames = 25
sampleMode = 'RU' # RU: Random Uniform, RP: Random Poisson, SF: Sequential Full, SI: Sequential Interval
gamma = 0.2

# Model inputs
trainAll = True

# Training inputs
batchSize = 8
minLR = 0.001
momentum = 0.9
weightDecay = 0.005
epochs = 100

saveName = 'nf{}_gm{}_cjAfter_sm{}_at{}_bs{}_lr{}_mm{}_wd{}_ep{}'.format(nFrames, gamma, sampleMode, trainAll, batchSize,
                                                                         minLR, momentum, weightDecay, epochs)

###################################
# Define Datasets and DataLoaders #
###################################

trainDataset = DarkHARDataset(stage='train', dataRoot=dataRoot, nClasses=nClasses, nFrames=nFrames, sampleMode=sampleMode, gamma=gamma)
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=0)
valDataset = DarkHARDataset(stage='validate', dataRoot=dataRoot, nClasses=nClasses, nFrames=nFrames, sampleMode=sampleMode, gamma=gamma)
valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=0)
print('Data Loaded', flush=True)

################
# Define Model #
################

# Load model with pretrained weights
model = r3d_18(weights=R3D_18_Weights.DEFAULT)

# If you don't want to train all parameters, then turn off requires_grad for existing parameters
if not(trainAll):
    for p in model.parameters(): p.requires_grad = False

# Custom bad code for freezing certain ResNet layers
# ResNet50 has 7 children: stem, layer1, layer2, layer3, layer4, avgpool, fc
# The Seq (layer#) blocks each contain basic blocks with each basic block containing 2-3 convolutions
#freezeN = 4 # Number of children to freeze (not 0-indexed, just straight up number)
#childCount = 0 # Initialize counter
#for child in model.children():
#    childCount += 1
#    if childCount > freezeN:
#        break
#    for param in child.parameters(): param.requires_grad = False

# Replace final classification layer
model.fc = torch.nn.Linear(512, 10)
model.cuda()
print('Model Loaded', flush=True)

# Count trainable and total model parameters
def count_params(model):
    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    allParams = sum(p.numel() for p in model.parameters())
    return trainableParams, allParams
trainableParams, allParams = count_params(model)
print('- Trainable Params: {}\n- Total Params: {}'.format(trainableParams, allParams))

#############################
# Training/Evaluation Setup #
#############################

# Loss function and optimizer
lossFcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=minLR, momentum=momentum, weight_decay=weightDecay, nesterov=True)

# Cosine learning rate decay from lrMax at step 0 to lrMin at step totalSteps
def cosine_annealing(step, totalSteps, lrMax, lrMin):
    return lrMin + (lrMax - lrMin) * 0.5 * (1 + np.cos(step / totalSteps * np.pi))

# Lambda function and learning rate scheduler method
lambda1 = lambda step: cosine_annealing(step, epochs * len(trainDataLoader), 1, minLR)
lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# Instantiate trainer from custom class
trainer = DarkHARTrainer(model, trainDataLoader=trainDataLoader, lossFcn=lossFcn, optimizer=optimizer, lrScheduler=lrScheduler)
print('Trainer Loaded', flush=True)

# Instantiate evaluator from custom class
evaluator = DarkHAREvaluator(model, valDataLoader=valDataLoader, lossFcn=lossFcn)
print('Evaluator Loaded', flush=True)

############################
# Training/Evaluation Loop #
############################

print('Running ' + saveName)

beginEpoch = time.time()

# Loop through epochs, train and evaluate, and print validation results
for epoch in range(0, epochs):
    trainLoss = trainer.train_epoch()
    print(time.time() - beginEpoch)
    evalLoss, evalAcc = evaluator.eval_epoch()

    print('{} | Train_Loss: {} | Val_Loss: {} | Val_Acc: {}'.format(epoch, trainLoss, evalLoss, evalAcc * 100))

torch.save(model.state_dict(), saveName + '_last.ckpt')

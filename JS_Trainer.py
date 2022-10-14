import torch
from tqdm import tqdm


class DarkHARTrainer:
    def __init__(self, model, trainDataLoader, lossFcn, optimizer, lrScheduler):
        self.model = model
        self.trainDataLoader = trainDataLoader
        self.lossFcn = lossFcn
        self.optimizer = optimizer
        self.lrScheduler = lrScheduler

    def train_epoch(self):
        # Set model in training mode
        self.model.train()

        # Initialize average training loss
        avgLoss = 0.0

        # Loop through batches
        for videoTens, labelTens in tqdm(self.trainDataLoader):
            # Get batched tensors and push to cuda
            videoTens = videoTens.cuda()
            labelTens = labelTens.cuda()

            # Forward computation and loss calculation
            predTens = self.model(videoTens).softmax(1)
            loss = self.lossFcn(predTens, labelTens)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lrScheduler.step()

            # Accumulate loss
            with torch.no_grad():
                avgLoss += loss.item()

        # Reweight average loss by all the batches
        avgLoss /= len(self.trainDataLoader)

        return avgLoss


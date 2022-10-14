import torch


class DarkHAREvaluator:
    def __init__(self, model, valDataLoader, lossFcn):
        self.model = model
        self.valDataLoader = valDataLoader
        self.lossFcn = lossFcn

    def eval_epoch(self):
        # Set model in evaluation mode
        self.model.eval()

        # Initialize average validation loss and accuracy
        avgLoss, accuracy = 0.0, 0

        # Loop through batches, no gradients attached to quantities
        with torch.no_grad():
            for videoTens, labelTens in self.valDataLoader:
                # Get batched tensors and push to cuda
                videoTens = videoTens.cuda()
                labelTens = labelTens.cuda()

                # Forward computation and loss calculation
                predTens = self.model(videoTens).softmax(1)
                loss = self.lossFcn(predTens, labelTens)

                # Accumulate validation loss and correct count
                avgLoss += loss.item()
                accuracy += (predTens.argmax(1) == labelTens.argmax(1)).type(torch.float).sum().item()

            # Reweight avgLoss by batches and accuracy by total number of samples
            avgLoss /= len(self.valDataLoader)
            accuracy /= len(self.valDataLoader.dataset)

        return avgLoss, accuracy


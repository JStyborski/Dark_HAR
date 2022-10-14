import torch
import torchvision
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from torchvision.models.feature_extraction import create_feature_extractor

# Instance the model, set in eval mode, and set the weights path
model1 = r3d_18()
model1.eval()
PATH1 = r'pth_from_pytorch\r3d_18-b3b3357e.pth'

# model2 = mc3_18()
# model2.eval()
# PATH2 = r'pth_from_pytorch\mc3_18-a90a0ba3.pth'

# model3 = r2plus1d_18()
# model3.eval()
# PATH3 = r'pth_from_pytorch\r2plus1d_18-91a641e6.pth'

device = torch.device('cpu')
model1.load_state_dict(torch.load(PATH1, map_location=device))

# Dictionary of the layers of interest for feature extraction from the model
# layers1-4 correspond to the 4 Sequential layers that contain blocks of 3D convolutions
# avgpool refers to the final AdaptiveAvgPool3d layer at the end, right before the final fc layer
return_nodes = {
	'layer1': 'layer1',
	'layer2': 'layer2',
	'layer3': 'layer3',
	'layer4': 'layer4',
	'avgpool': 'avgpool',
}

# Some input tensor representing an input video
video = torch.randn(1,3,16,112,112) # Input video with: Batch size, channels, number of frames, height, width

# Creates a feature extractor object from the model for the given layers of interest
feat_extract = create_feature_extractor(model1, return_nodes=return_nodes)

# Returns the tensor corresponding to [samples, channels, frames, height, width] for the given layer
feature = feat_extract(video)['avgpool'].squeeze(-1).squeeze(-1).squeeze(-1) # Feature obtained after pooling

# The original model has a final FC layer that outputs 400 classes
# Push the input video through the model and softmax the output
pred = model1(video).squeeze(0).softmax(0) # Prediction with the original 400 classes (can be viewed as another kind of feature)

# Get maximum of sotmax and output
# label = pred.argmax().item() # For checking only
# print("Label: {}".format(label))
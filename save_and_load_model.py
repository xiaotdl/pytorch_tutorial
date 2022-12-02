"""
Save and Load the Model
source: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
"""


# In this section we will look at how to persist ** model state ** with 
# 1) saving, 
# 2) loading and 
# 3) running model predictions.

import torch
import torchvision.models as models

print("== Saving and Loading Model Weights ==")


# saving the model
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# loading the model
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
# Sets the module in evaluation mode.
# This is equivalent with self.train(False).
model.eval()

# Note: 
# be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.


print("== Saving and Loading Models with Shapes ==")


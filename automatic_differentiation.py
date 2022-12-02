"""
Automatic Differentiation with torch.autograd
source: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
"""

# When training neural networks, the most frequently used algorithm is back propagation.
# In this algorithm, parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter.

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


print("== Tensors, Functions and Computational graph ==")

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


print("== Computing Gradients ==")

loss.backward()
print(w.grad)
print(b.grad)


print("== Disabling Gradient Tracking ==")

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

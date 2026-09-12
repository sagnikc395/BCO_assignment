## trainging in torch

import torch
from torchvision.models import resnet18, ResNet18_Weights

# set model and weights and data and labels
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

## run the input data through the model through each of its layers to make a prediction.
# also called as a forward pass.

prediction = model(data)

## using the models predictions and the corresponding label to calculate the error.
# and the next step is to backpropogate this error through the network.
# backprop is kicked off when we call .backward() on the error tensor.
# autograd will then calculate and store the gradient for each model parameter in the parameter's
# .grad attribute

loss = (prediction - labels).sum()
loss.backward()  # backward pass

## load an optimizer, SGD with lr=0.01 and momentum of 0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# init step to initiate gradient descent
optim.step()

# creating two tensors a and b with requires_grad=True to signal to autograd the every operation on them should be tracked.

a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

Q = 3 * a**3 - b**2

external_grad = torch.tensor([1.0, 1.0])
Q.backward(gradient=external_grad)

# gradients now updates in a.grad and b.grad
print(a.grad)
print(b.grad)

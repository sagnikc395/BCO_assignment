# What is PyTorch ?

- a replacement for Numpy to use the power of GPUs and other accelerators.
- an automatic differentiation library that is useful to implement neural networks.

## Goal:
- Understsand torch's tensor library and neural networks at a higher level.
- train a smaller neural network to classify images.

## Tensors
- Specialized data structure that are very similar to arrays and matrices
- we use tensors to encode the inputs and outputs of a model, as well as the model's parameters.
- similar to numpy's ndarrays, except that tensors can run on GPUs or other specialized hardware to accelerate computing.
- they can be initialized in various ways, including creating directly from data, the data type is automatically inferred.
- shape is a tuple of tensor dimensions. It determines the dimensionality of the output tensor.

- tensor attributes:
  - describe their shape, datatype and the device the tensors are stored.

- can do tensor mathematical operations on the GPU by selecting the appropriate GPU type.
  - ops like add, subtract, divide , matmul, mul etc.

## Gentle Intro to torch.autograd
- `torch.autograd` is torch's automatic differentiation engine that powers neural networks
training.
- NNs are a collection of nested functions that are executed on some input data.
- these functions are defined by parameters (consisting of weights and biases) , which in pytorch are stored as tensors.
- training a NN happenes in 2 stages.
  - Forward Propogation -> in forward prop, the NN makes its best guess about the correct output. It will run the input data
  through each of its functions to make this guess.
  - Backward Propogation -> In backprop, NN will adjust its parameters proportionate to the error in its guess. It will do this by traversing
  backwards from the output, collecting the derivaties of the error wrt to the parameters of the functions (gradients), and optimizing the parameters
  using gradient descent.


## Neural Networks
- NN can be constructed using the `torch.nn` package.
- Typical training procedure for a neural network is as follows:
  - Define the neural network that has some learnable parameters(or weights)
  - iterate over a dataset of inputs
  - process input through the network
  - compute the loss (how far is the output from being correct)
  - Propogate gradients back into the networks parameters
  - update the weights of the network, typically using a simple update rule:
  `weight = weight - learning_rate * gradient`
  - after defining the `forward` and the `backward` functions( where the gradients are computed)
  , is automatically defined by using `autograd`. We can use any of the Tensor operations in the `forward` fuction.

- loss function takes the (output,target) pair of inputs, and computes a vlaue that estimates how far away the output is from the target.
- several different loss functions under the nn package.
- backprop:
  - to backpropogate the error all we have to do is to do `loss.backward()`
  - need to clear the existing gradients though , else the gradients will be accumulated to exsiting gradients.

- 

## Training a Classifier




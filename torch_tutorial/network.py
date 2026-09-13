import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. input image channel, 6 output channels, 5*5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # affine operation: y = Wk + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))

        s2 = F.max_pool2d(c1, (2, 2))

        c3 = F.relu(self.conv2(s2))

        s4 = F.max_pool2d(c3, 2)

        s4 = torch.flatten(s4, 1)

        f5 = F.relu(self.fc1(s4))

        f6 = F.relu(self.fc2(f5))

        output = self.fc3(f6)
        return output


if __name__ == "__main__":
    net = Network()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    # a random 32*32 input
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    # zero the gradient buffers of all parameters and backprops with random gradients
    net.zero_grad()
    out.backward(torch.randn(1, 10))

    # loss
    output = net(input)
    target = torch.randn(10)  # dummy target
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    net.zero_grad()  # zeros the gradient buffers of all parameters

    print("conv1.bias.grad before backward")
    print(net.conv1.bias.grad)

    loss.backward()

    print("conv1.bias.grad after backward")
    print(net.conv1.bias.grad)

    # now updating the weights
    learning_rate = 0.01
    for f in net.parameters():
        with torch.no_grad():
            f -= f.grad * learning_rate

    # using optimizer with SGD
    import torch.optim as optim

    # creating the optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # training loop
    optimizer.zero_grad()  # zeros the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    # does the update
    optimizer.step()

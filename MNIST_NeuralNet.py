import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 400)
        self.fc2 = torch.nn.Linear(400, 400)
        self.fc3 = torch.nn.Linear(400, 200)
        self.fc4 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.log_softmax(x)


class NetWrapper:
    def __init__(self):
        self.net = Net()
        if torch.cuda.is_available():
            self.net.to('cuda:0')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.train_load = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.test_load = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    def learn(self, learning_rate, momentum, n_epochs, print_interval):
        optimizer = torch.optim.SGD(self.net.parameters(), learning_rate, momentum)
        criterion = torch.nn.NLLLoss()
        for epoch in range(n_epochs):
            for batch_idx, (data, target) in enumerate(self.train_load):
                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
                data = data.view(-1, 28 * 28)
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    net_out = self.net(data.to('cuda:0'))
                    loss = criterion(net_out, target.to('cuda:0'))
                else:
                    net_out = self.net(data)
                    loss = criterion(net_out, target)
                loss.backward()
                optimizer.step()
                if batch_idx % print_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                          .format(epoch, batch_idx * len(data), len(self.train_load.dataset),
                                  100. * batch_idx / len(self.train_load), loss.data.item()))

    def test(self):
        n_correct = 0
        n_incorrect = 0
        for batch_idx, (data, target) in enumerate(self.test_load):
            data = data.view(-1, 28 * 28)
            if torch.cuda.is_available():
                net_out = self.net(data.to('cuda:0'))
            else:
                net_out = self.net(data)

            if np.argmax(net_out.cpu().detach().numpy()) == target[0]:
                n_correct += 1
            else:
                n_incorrect += 1
        print(n_correct, "/", n_correct + n_incorrect)

    def use(self, image_matrix):
        tensor_input = self.transform(image_matrix).float()
        tensor_input = tensor_input.view(-1, 28 * 28)
        prob_matrix = self.net.forward(tensor_input)
        return prob_matrix.detach().numpy()


MNIST_net = NetWrapper()
MNIST_net.learn(0.07, 0.6, 20, 5)
MNIST_net.test()





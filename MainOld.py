import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

asdasdasdasd

device = torch.device('cuda')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 300)
        self.fc2 = torch.nn.Linear(300, 300)
        self.fc3 = torch.nn.Linear(300, 200)
        self.fc4 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.log_softmax(x)

    def read(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.log_softmax(x)




n_epochs = 5
log_interval = 1





def test(test_loader, net):
    n_correct = 0
    n_wrong = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.view(-1, 28*28)
        net_out = net(data)
        net_out = net_out.detach().numpy()
        net_val = np.argmax(net_out)
        act_val = target[0]
        if net_val == act_val:
            n_correct += 1
        else:
            n_wrong += 1
    return n_correct, n_wrong


n_c, n_w = test(test_loader, net)
print(n_c, '/', n_w + n_c, 'correct')
print("percent right: {}%".format(n_c / (n_c + n_w) * 100))


def use_net(image_matrix):
    tensor_input = transform(image_matrix)
    tensor_input = tensor_input.float()
    tensor_input = tensor_input.view(-1, 28*28)
    probs = net.read(tensor_input)
    return probs.detach().numpy()


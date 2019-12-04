import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time


class Net(torch.nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 200)
        self.fc4 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class MNISTData:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.train_load = torch.utils.data.DataLoader(train_set, batch_size=60, shuffle=True, num_workers=0)

        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.test_load = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)


class NetWrapper:
    def __init__(self):
        self.net = Net(28*28)
        if torch.cuda.is_available():
            self.net.to('cuda:0')

    def learn(self, train_loaded, batc_s, learning_rate, momentum, n_epochs):
        optimizer = torch.optim.SGD(self.net.parameters(), learning_rate, momentum)
        criterion = torch.nn.NLLLoss()
        print(f'\nNeural networking training with {n_epochs} epochs:')
        for epoch in range(n_epochs):
            s_time = time.time()
            loss_total = 0
            for batch_idx, (data, target) in enumerate(train_loaded):
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
                loss_total += loss
                leng = len(train_loaded)*batc_s
                n_blocks = int(batch_idx // (leng/batc_s/20))
                space = " "
                bar = u'\u2588'
                if epoch < 9:
                    print(f'\rEpoch {epoch+1}  |{bar*n_blocks}{space*(20-n_blocks)}| {batch_idx*batc_s}/{leng}', end='')
                else:
                    print(f'\rEpoch {epoch+1} |{bar*n_blocks}{space*(20-n_blocks)}| {batch_idx*batc_s}/{leng}', end='')
            if epoch < 9:
                print(f'\rEpoch {epoch + 1}  |{bar * 20}| {leng}/{leng}', end='')
                print(f'   {(time.time() - s_time):.2f}s  Avg Loss: {(loss_total / (leng/batc_s)):.4f}')
            else:
                print(f'\rEpoch {epoch + 1} |{bar * 20}| {leng}/{leng}', end='')
                print(f'   {(time.time() - s_time):.2f}s  Avg Loss: {(loss_total / (leng/batc_s)):.4f}')
        return loss_total / (leng/batc_s)

    def test(self, test_loaded):
        n_correct = 0
        n_incorrect = 0
        for batch_idx, (data, target) in enumerate(test_loaded):
            data = data.view(-1, 28 * 28)
            if torch.cuda.is_available():
                net_out = self.net(data.to('cuda:0'))
            else:
                net_out = self.net(data)
            if np.argmax(net_out.cpu().detach().numpy()) == target[0]:
                n_correct += 1
            else:
                n_incorrect += 1
            percent_correct = n_correct/(n_incorrect+n_correct)*100
            print(f'\r{batch_idx+1}/{len(test_loaded)} tests complete, {percent_correct:.2f}% correct', end='')
        print(f'')
        return percent_correct

    def use(self, image_matrix, transform):
        tensor_input = transform(image_matrix).float()
        tensor_input = tensor_input.view(-1, 28 * 28)
        prob_matrix = self.net.forward(tensor_input)
        return prob_matrix.detach().numpy()

    def save(self, filename):
        path = f'./neural_net/{filename}.pth'
        torch.save(self.net, path)

    def load(self, filename):
        path = f'./neural_net/{filename}.pth'
        if torch.cuda.is_available():
            self.net = torch.load(path, 'cuda:0')
        else:
            self.net = torch.load(path, 'cpu')

    def reset(self):
        self.net = None
        self.net = Net(28 * 28)
        if torch.cuda.is_available():
            self.net.to('cuda:0')

    def recorded_learn(self, data):
        error_matrix = np.zeros((3, 6, 5))
        for epx, epochs in enumerate([20, 25, 30]):
            for mdx, momentum in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
                for lrx, learning_rate in enumerate([0.001, 0.002, 0.004, 0.006, 0.008, 0.01]):
                    error = 0
                    for erx, era in enumerate([1, 2, 3]):
                        self.reset()
                        self.learn(data.train_load, 60, learning_rate, momentum, epochs)
                        error = self.test(data.test_load) + error
                    error = round((error / 3), 2)
                    error_matrix[epx, lrx, mdx] = error
                    print(f'momentum: {momentum} learning rate: {learning_rate}, at {epochs} epochs, 3-avg error was {error:.2f}')
        print(error_matrix)










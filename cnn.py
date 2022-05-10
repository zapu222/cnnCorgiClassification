import torch

"""
class CNN(torch.nn.Module):
    def __init__(self, keep_prob=0.95):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob))

        self.fc1 = torch.nn.Linear(18 * 18 * 128, 512, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = torch.nn.Linear(512, 256, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) 

        self.fc3 = torch.nn.Linear(256, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
"""

class CNN(torch.nn.Module):
    def __init__(self, keep_prob=0.95):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=10, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob))

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob))

        self.fc1 = torch.nn.Linear(8 * 8 * 256, 512, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = torch.nn.Linear(512, 256, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) 

        self.fc3 = torch.nn.Linear(256, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
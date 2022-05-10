import json
import torch
import argparse

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from cnn import CNN
from dataset import CorgiImages


def train(args):
    train_path, test_path, save_path, lr, epochs, device = \
        args['train_path'], args['test_path'], args['save_path'], args['lr'], args['epochs'], args['device']

    trainset = CorgiImages(train_path)
    testset = CorgiImages(test_path, test=True)

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=40, shuffle=True)

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1} of {epochs}")
        running_loss = []

        for _, data in enumerate(tqdm(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        print(f"Average Loss: {sum(running_loss) / len(running_loss)}")

        torch.save(model.state_dict(), save_path)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)

                _, labels = torch.max(labels.squeeze().data, 1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on {total} test images: {100 * correct // total} %\n")

    print('Finished Training')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, default='', help='train_args.json path')

    opt = parser.parse_args()
    with open(opt.args_path, 'r') as f:
        args = json.load(f)
    with open(args['save_path'][0:-3] + 'json', 'w') as g:
        json.dump(args, g, indent=2)
    return args


if __name__ == "__main__":
    args = parse_opt()
    train(args)
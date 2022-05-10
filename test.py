import json
import torch
import argparse

from torch.utils.data import DataLoader

from cnn import CNN
from dataset import CorgiImages


def test(args):
    print('\n----------------------------------------------------------------------------------------------------\n')
    model_path, test_path, device = \
        args['model_path'], args['test_path'], args['device']

    testset = CorgiImages(test_path, test=True)
    testloader = DataLoader(testset, batch_size=40, shuffle=True)

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    print('Model loaded correctly; model summary...\n')
    print(model)

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

    print(f"\nAccuracy on {total} test images: {100 * correct // total} %")
    print('\n----------------------------------------------------------------------------------------------------\n')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, default='', help='test_args.json path')

    opt = parser.parse_args()
    with open(opt.args_path, 'r') as f:
        args = json.load(f)
    return args


if __name__ == "__main__":
    args = parse_opt()
    test(args)
import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils import augment_image


class CorgiImages(Dataset):
    def __init__(self, path, test=False):
        self.test = test

        self.labels = {"cardigan": 0, "pembroke": 1}

        self.corgis = []
        folders = os.listdir(path)

        for img in os.listdir(os.path.join(path, folders[0])):
            self.corgis.append([folders[0], os.path.join(path, folders[0], img)])

        for img in os.listdir(os.path.join(path, folders[1])):
            self.corgis.append([folders[1], os.path.join(path, folders[1], img)])

        random.shuffle(self.corgis)

    def __len__(self):
        return len(self.corgis)

    def __getitem__(self, idx):
        img_path = self.corgis[idx][1]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (320, 320),0,0, cv2.INTER_LINEAR)

        if self.test != True:
            image = augment_image(image)
        
        transform = ToTensor()
        x = transform(image)

        label = self.corgis[idx][0]
        label = self.labels[label]

        y = np.zeros((1,len(self.labels)))
        for i, _ in enumerate(x[0]):
            if i == label:
                y[0][i] = 1

        return x, y
import numpy as np
import torch

import os
dirname = os.path.dirname(__file__)


class TinyImages(torch.utils.data.Dataset):

    def __init__(self, root='/home/scratch/datasets/80M Tiny Images/tiny_images.bin', transform=None, exclude_cifar=True):

        data_file = open(root, "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape((32, 32, 3), order="F")

        self.load_image = load_image
        self.offset = 0     # offset index
        self.length = 79302017
        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open(os.path.join(dirname, '80mn_cifar_idxs.txt'), 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

            # bisection search option
            # self.cifar_idxs = tuple(sorted(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar: #rewritten to make it always deterministic
            resample_step = 100000
            i = 0
            index0 = index
            while self.in_cifar(index):
                index = (index + resample_step) % 79302016
                i += 1
                if i == 100:
                    resample_step += 1
                    i = 0
                    index = index0

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return self.length

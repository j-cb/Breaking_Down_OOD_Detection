import torchvision
import os
import numpy as np

class OpenImages(torchvision.datasets.ImageFolder):
    def __init__(self, split, root='/home/scratch/datasets/openimages', transform=None, target_transform=None, exclude_dataset=None):
        if split == 'train':
            path = os.path.join(root, 'train')
        elif split == 'val':
            path = os.path.join(root, 'val')
        elif split == 'test':
            raise NotImplementedError()
            path = os.path.join(root, 'test')
        else:
            raise ValueError()

        super().__init__(path, transform=transform, target_transform=target_transform)
        exclude_idcs = []

        if exclude_dataset is not None and split == 'train':
            if exclude_dataset == 'imageNet100':
                duplicate_file = 'openImages_imageNet100_duplicates.txt'
            elif exclude_dataset == 'flowers':
                duplicate_file = 'utils/openImages_flowers_idxs.txt'
            elif exclude_dataset == 'pets':
                duplicate_file = 'utils/openImages_pets_idxs.txt'
            elif exclude_dataset == 'cars':
                duplicate_file = 'utils/openImages_cars_idxs.txt'
            elif exclude_dataset == 'food-101':
                duplicate_file = 'utils/openImages_food-101_idxs.txt'
            elif exclude_dataset == 'cifar10':
                print('Warning; CIFAR10 duplicates not checked')
                duplicate_file =  None
            else:
                raise ValueError(f'Exclusion dataset {exclude_dataset} not supported')

            if duplicate_file is not None:
                with open(duplicate_file, 'r') as idxs:
                    for idx in idxs:
                        exclude_idcs.append(int(idx))

        self.exclude_idcs = set(exclude_idcs)
        print(f'OpenImages {split} - Length: {len(self)} - Exclude images: {len(self.exclude_idcs)}')

    def __getitem__(self, index):
        while index in self.exclude_idcs:
            index = np.random.randint(len(self))

        return super().__getitem__(index)[0], 0

class OpenImages32(torchvision.datasets.ImageFolder):
    def __init__(self, split, root='/home/scratch/datasets/openimages32', transform=None, target_transform=None, exclude_dataset=None):
        if split == 'train':
            path = os.path.join(root, 'train')
        elif split == 'val':
            path = os.path.join(root, 'train')
        elif split == 'test':
            raise NotImplementedError()
            #path = os.path.join(root, 'test')
        else:
            raise ValueError()

        super().__init__(path, transform=transform, target_transform=target_transform)
        exclude_idcs = []

        if exclude_dataset is not None and split == 'train':
            if exclude_dataset == 'imageNet100':
                duplicate_file = 'openImages_imageNet100_duplicates.txt'
            elif exclude_dataset == 'flowers':
                duplicate_file = 'utils/openImages_flowers_idxs.txt'
            elif exclude_dataset == 'pets':
                duplicate_file = 'utils/openImages_pets_idxs.txt'
            elif exclude_dataset == 'cars':
                duplicate_file = 'utils/openImages_cars_idxs.txt'
            elif exclude_dataset == 'food-101':
                duplicate_file = 'utils/openImages_food-101_idxs.txt'
            elif exclude_dataset == 'cifar10':
                print('Warning; CIFAR10 duplicates not checked')
                duplicate_file =  None
            else:
                raise ValueError(f'Exclusion dataset {exclude_dataset} not supported')

            if duplicate_file is not None:
                with open(duplicate_file, 'r') as idxs:
                    for idx in idxs:
                        exclude_idcs.append(int(idx))

        self.exclude_idcs = set(exclude_idcs)
        print(f'OpenImages {split} - Length: {len(self)} - Exclude images: {len(self.exclude_idcs)}')

    def __getitem__(self, index):
        while index in self.exclude_idcs:
            index = np.random.randint(len(self))

        return super().__getitem__(index)[0], 0


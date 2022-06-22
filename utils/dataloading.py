import warnings

import torch
from torchvision import datasets, transforms

import utils.noisefunctions as noisefunctions
import utils.paths_config as paths_config
from utils.autoaugment import CIFAR10Policy
from utils.openimages_loader import OpenImages, OpenImages32
from utils.tinyimages_80mn_loader import TinyImages


class Pert_Noise_Dataset(torch.utils.data.dataset.Dataset):
    """A dataset that is built from a ground dataset and a noise function, returning noisy images.py of the same shape as
    the ground data.
       noise_fn should be a function accepting a ground data sample with its label and returning the noisy sample and
       label with the same shape as the input
    """

    def __init__(self, ground_ds, noise_fn, label_fn=None, transform=None):
        self.ground_ds = ground_ds
        self.noise_fn = noise_fn
        self.label_fn = label_fn
        self.transform = transform
        # self.__name__ = noise.__name__ + '_on_' + ground_ds.__name__

    def __getitem__(self, index):
        index %= len(self.ground_ds)
        noisy = self.noise_fn(self.ground_ds[index])
        inp = noisy[0]
        if self.transform is not None:
            inp = transforms.ToPILImage()(inp)
            inp = self.transform(inp)

        if self.label_fn is None:
            lbl = noisy[1]
        else:
            lbl = self.label_fn(noisy[1])
        return inp, lbl

    def __len__(self):
        return 1000000  # len(self.ground_ds)

    _repr_indent = 4

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__ + '_' + self.noise_fn.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


class Noise_Dataset(torch.utils.data.dataset.Dataset):
    """A dataset that is built from a ground dataset and a noise function, returning noisy images.py of the same shape
    as the ground data. noise_fn should be a function accepting a ground data sample with its label and returning the
    noisy sample and label with the same shape as the input
    """

    def __init__(self, ground_ds, noise_fn, label_fn=None, transform=None):
        self.noise_fn = noise_fn
        self.label_fn = label_fn
        self.transform = transform
        # self.__name__ = noise.__name__ + '_on_' + ground_ds.__name__
        self.likesample = ground_ds[0]

    def __getitem__(self, index):
        noisy = self.noise_fn(self.likesample)
        inp = noisy[0]
        if self.transform is not None:
            inp = transforms.ToPILImage()(inp)
            inp = self.transform(inp)

        if self.label_fn is None:
            lbl = noisy[1]
        else:
            lbl = self.label_fn(noisy[1])
        return inp, lbl

    def __len__(self):
        return 1000000

    _repr_indent = 4

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__ + '_' + self.noise_fn.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


# noinspection PyTypeChecker
def getloader_TINY(train, batch_size, augmentation, dataloader_kwargs, exclude_cifar=True, shuffle=None):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            []
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )

    tiny_dset = TinyImages(transform=transform, exclude_cifar=exclude_cifar)
    if train:
        dset = torch.utils.data.Subset(tiny_dset, range(100000, 50000000))
        dset.length = len(range(100000, 50000000))
        dset.__repr__ = tiny_dset.__repr__
    else:
        dset = torch.utils.data.Subset(tiny_dset, range(50000000, 50000000 + 40000))
        dset.length = len(range(50000000, 50000000 + 40000))
        dset.__repr__ = tiny_dset.__repr__
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle if shuffle is not None else train,
        **dataloader_kwargs)
    return loader


# noinspection PyTypeChecker
def getloader_OpenImages(train, batch_size, augmentation, dataloader_kwargs, exclude_dataset=None, shuffle=None):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            [transforms.transforms.Resize((32, 32)), ]
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )
    if train:
        split = 'train'
    else:
        split = 'val'
    open_dset = OpenImages(split, paths_config.location_dict['OpenImages'], transform=transform)
    if split == 'train':
        dset = open_dset
    elif split == 'val':
        dset = torch.utils.data.Subset(open_dset, range(10000))
        dset.__repr__ = open_dset.__repr__
    else:
        raise ValueError(f'Split {split} not available.')
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle if shuffle is not None else train,
        **dataloader_kwargs)
    return loader

def getloader_OpenImages32(train, batch_size, augmentation, dataloader_kwargs, exclude_dataset=None, shuffle=None):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            [transforms.transforms.Resize((32, 32)), ]
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )
    if train:
        split = 'train'
    else:
        split = 'val'
    open_dset = OpenImages32(split, paths_config.location_dict['OpenImages32'], transform=transform)
    if split == 'train':
        dset = open_dset
    elif split == 'val':
        dset = torch.utils.data.Subset(open_dset, range(10000))
        dset.__repr__ = open_dset.__repr__
    else:
        raise ValueError(f'Split {split} not available.')
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle if shuffle is not None else train,
        **dataloader_kwargs)
    return loader


def getloader_CIFAR10(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            []
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )

    dset = datasets.CIFAR10(paths_config.location_dict['CIFAR10'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_CIFAR100(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            []
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )

    dset = datasets.CIFAR100(paths_config.location_dict['CIFAR100'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_SVHN(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        if augmentation.get('HFlip'):
            warnings.warn(
                f'Random horizontal flip augmentation selected for SVHN, which usually is not done. Augementations: {augmentation}')
        transform = transforms.Compose(
            []
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )
    if train:
        split = 'train'
    else:
        split = 'test'
    dset = datasets.SVHN(paths_config.location_dict['SVHN'], split=split, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noise_fn, pert=False):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            []
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )
    if augmentation.get('28g'):
        ground_ds = datasets.MNIST(paths_config.location_dict['MNIST'], train=train, download=True,
                                   transform=transforms.ToTensor())
    else:
        ground_ds = datasets.CIFAR10(paths_config.location_dict['CIFAR10'], train=train, download=True,
                                     transform=transforms.ToTensor())

    if pert:
        dset = Pert_Noise_Dataset(ground_ds, noise_fn, label_fn=lambda x: 0, transform=transform)
    else:
        dset = Noise_Dataset(ground_ds, noise_fn, label_fn=lambda x: 0, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_Uniform(train, batch_size, augmentation, dataloader_kwargs):
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.noise_uniform)


def getloader_Smooth(train, batch_size, augmentation, dataloader_kwargs):
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.noise_low_freq)


def getloader_Black(train, batch_size, augmentation, dataloader_kwargs):
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.monochrome(0))


def getloader_Likelihood_Ratios_Noisy(train, batch_size, augmentation, dataloader_kwargs, mu=0.1):
    warnings.warn('The Likelihood_Ratios_Noisy dataset is based on CIFAR-10 or MNIST depending on size.')
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.lh_ratios_style(mu),
                           pert=True)


def getloader_LSUN_CR(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            [transforms.Resize(size=(32, 32)), ]
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )
    if train:
        raise ValueError(
            f'Only the validation split of LSUN Classroom is available. train is set to {train}, which is not allowed.')
    else:
        classes = ['classroom_val']
    dset = datasets.LSUN(paths_config.location_dict['LSUN'], classes=classes, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_LSUN(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            [transforms.Resize(size=(32, 32)), ]
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )
    if train:
        raise ValueError(
            f'Only the validation split of LSUN Classroom is available. train is set to {train}, which is not allowed.')
    else:
        classes = 'val'
    dset = datasets.LSUN(paths_config.location_dict['LSUN'], classes=classes, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_CelebA(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'):  # MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(
                f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.Grayscale(),
            transforms.RandomCrop(28, augmentation.get('crop', 0)),
            transforms.ToTensor(),
        ])
    else:  # stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
            [transforms.Resize(size=(32, 32)), ]
            + augmentation.get('HFlip', False) * [transforms.RandomHorizontalFlip(), ]
            + bool(augmentation.get('crop', 0)) * [transforms.RandomCrop(32, augmentation.get('crop', 0)), ]
            + augmentation.get('autoaugment', False) * [CIFAR10Policy(), ]
            + [transforms.ToTensor(), ]
        )
    if train:
        raise NotImplementedError
    else:
        split = 'test'
    dset = datasets.CelebA(paths_config.location_dict['CelebA'], split=split, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


datasets_dict = {'TINY': getloader_TINY,
                 'OpenImages': getloader_OpenImages,
                 'OpenImages32': getloader_OpenImages32,
                 'CIFAR10': getloader_CIFAR10,
                 'CIFAR100': getloader_CIFAR100,
                 'SVHN': getloader_SVHN,
                 'Uniform': getloader_Uniform,
                 'Smooth': getloader_Smooth,
                 'Black': getloader_Black,
                 'LSUN_CR': getloader_LSUN_CR,
                 'LSUN': getloader_LSUN,
                 'CelebA': getloader_CelebA,
                 'Lh_Ratios': getloader_Likelihood_Ratios_Noisy
                 }


def get_test_out_loaders(batch_size, dataloader_kwargs, test_loaders_dict):
    out_loaders_dict = dict([])
    for name, augm in test_loaders_dict.items():
        try:
            out_loaders_dict[name] = datasets_dict[name](train=False, batch_size=batch_size, augmentation=augm,
                                      dataloader_kwargs=dataloader_kwargs)
        except Exception as e:
            print(f'{name} not loaded: {getattr(e, "message", repr(e))}')
            out_loaders_dict[name] = None
    return out_loaders_dict


num_classes = {'CIFAR10': 10, 'CIFAR100': 100}

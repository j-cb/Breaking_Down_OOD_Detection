# based on code from https://github.com/hendrycks/outlier-exposure [OE]
import argparse
import datetime
import json

import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.utils

from models.wrn import WideResNet

import utils.specs as specs
from utils.tinyimages_80mn_loader import TinyImages
from utils.openimages_loader import OpenImages, OpenImages32
from utils.tiny100kopenimages_loader import Tiny100kOpenImages
from utils.autoaugment import CIFAR10Policy
from utils.cutout import Cutout
import utils.default_args as default_args
import importlib

parser = argparse.ArgumentParser(description='Train a model for OOD detection.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Required arguments (not positional)
parser.add_argument('--dataset_in', type=str, choices=['CIFAR10', 'CIFAR100'],
                    help='Choose between CIFAR-10, CIFAR-100.', required=True)
parser.add_argument('--dataset_out', type=str, choices=['TINY', 'OpenImages',  'OpenImages32', 'SVHN', 'Tiny100kOpenImages'],
                    help='Choose between TINY and OpenImages.', required=True)
parser.add_argument('--method', type=str, choices=['Plain', 'OE', 'BGC', 'SharedCD', 'SepD', 'InMultiClassBin',
                                                   'SharedInMultiClassBin'],
                    help='Choose the training method which determines output neurons (with dataset) and loss.',
                    required=True)
parser.add_argument('--save', type=str, help='Folder to save checkpoints.', required=True)

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--in_batch_size', '-ib', type=int, default=128, help='Batch size in-distribution.')
parser.add_argument('--out_batch_size', '-ob', type=int, default=256, help='Batch size out-distribution.')
parser.add_argument('--out_data_number_limit', type=int, default=-1, help='Restrict number of available OOD samples.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--augmentation', type=str, default='standard',
                    choices=['standard', 'autoaugment', 'autoaugment_only_out'],
                    help='autoaugment cutout can be turned off with --AA_no_cutout')
parser.add_argument('--AA_cutout', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='Does autoaugment include cutout?')
parser.add_argument('--lamb', type=float, default=1.0)
parser.add_argument('--pass_all', action='store_true', help='pass unused batch for separate/plain')

# WRN Architecture
parser.add_argument('--architecture_base', '-ab', type=str, default='wrn', help='Choose architecture.', choices=['wrn'])
parser.add_argument('--layers', '-al', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', '-aw', default=2, type=int, help='widen factor wrn')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use for single gpu')
parser.add_argument('--num_workers', type=int, default=4, help='Pre-fetching threads.')
# Naming
parser.add_argument('--show_args', nargs='*', help='include these hyperparameters is model strings')
# Machine specific
parser.add_argument('--paths_file', type=str, default='utils.paths_config',
                    choices=['utils.paths_config', 'utils.paths_slurm'])
args = parser.parse_args()
args_dict = vars(args)
if args.show_args is None:
    args.show_args = []
paths_config = importlib.import_module(args.paths_file)


print(args_dict)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.ngpu > 0:
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True  # can improve speed of convolutions

if args.augmentation == 'standard':  # same transformation as [OE]
    train_transform = trn.Compose([trn.RandomHorizontalFlip(),
                                   trn.RandomCrop(32, padding=4),
                                   trn.ToTensor(),
                                   trn.Normalize(specs.cifar_mean, specs.cifar_std)])
    train_transform_out = train_transform
elif args.augmentation == 'autoaugment':
    train_transform = trn.Compose(([trn.RandomHorizontalFlip(),
                                   trn.RandomCrop(32, padding=4),
                                   CIFAR10Policy(),
                                   trn.ToTensor()] +
                                   args.AA_cutout * [Cutout(1, 16)] +
                                   [trn.Normalize(specs.cifar_mean, specs.cifar_std)]))
    train_transform_out = train_transform
elif args.augmentation == 'autoaugment_only_out':  # same transformation as [OE]
    train_transform = trn.Compose([trn.RandomHorizontalFlip(),
                                   trn.RandomCrop(32, padding=4),
                                   trn.ToTensor(),
                                   trn.Normalize(specs.cifar_mean, specs.cifar_std)])
    train_transform_out = trn.Compose(([trn.RandomHorizontalFlip(),
                                   trn.RandomCrop(32, padding=4),
                                   CIFAR10Policy(),
                                   trn.ToTensor()] +
                                   args.AA_cutout * [Cutout(1, 16)] +
                                   [trn.Normalize(specs.cifar_mean, specs.cifar_std)]))
else:
    raise NotImplementedError(f'Augmentation {args.augmentation}')

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(specs.cifar_mean, specs.cifar_std)])

if args.dataset_in == 'CIFAR10':
    train_data_in = dset.CIFAR10(paths_config.location_dict['CIFAR10'], train=True, transform=train_transform)
    test_data_in = dset.CIFAR10(paths_config.location_dict['CIFAR10'], train=False, transform=test_transform)
elif args.dataset_in == 'CIFAR100':
    train_data_in = dset.CIFAR100(paths_config.location_dict['CIFAR100'], train=True, transform=train_transform)
    test_data_in = dset.CIFAR100(paths_config.location_dict['CIFAR100'], train=False, transform=test_transform)
else:
    raise NotImplementedError(f'dataset_in {args.dataset_in}')
num_in_classes = specs.get_num_in_classes(args.dataset_in)

ood_dataset_dict = {'TINY': TinyImages,
                    'OpenImages': OpenImages,
                    'OpenImages32': OpenImages32,
                    'Tiny100kOpenImages': Tiny100kOpenImages,
                    'SVHN': dset.SVHN
                    }
pre_transform_dict = {'TINY': trn.Compose([trn.ToTensor(), trn.ToPILImage()]),
                      'OpenImages': trn.Compose([trn.ToTensor(), trn.ToPILImage(), trn.Resize((32, 32))]),
                      'OpenImages32': trn.Compose([trn.ToTensor(), trn.ToPILImage()]),
                      'Tiny100kOpenImages': trn.Compose([trn.ToTensor(), trn.ToPILImage()]),
                      'SVHN': trn.Compose([]),
                      }
train_shuffle_dict = {
    'TINY': False,
    'OpenImages': True,
    'OpenImages32': True,
    'Tiny100kOpenImages': True,
    'SVHN': True,
}

if args.dataset_out in {'TINY'}:
    ood_data = ood_dataset_dict[args.dataset_out](root=paths_config.location_dict['TinyImages'],
                                                  transform=trn.Compose([pre_transform_dict[args.dataset_out],
                                                                     train_transform_out]))
elif args.dataset_out in {'OpenImages'}:
    ood_data = ood_dataset_dict[args.dataset_out](root=paths_config.location_dict['OpenImages'],
                                                  split='train',
                                                  transform=trn.Compose([pre_transform_dict[args.dataset_out],
                                                                     train_transform_out]))
elif args.dataset_out in {'Tiny100kOpenImages', 'OpenImages32'}:
    ood_data = ood_dataset_dict[args.dataset_out](root=paths_config.location_dict[args.dataset_out],
                                                  split='train',
                                                  transform=trn.Compose([pre_transform_dict[args.dataset_out],
                                                  train_transform_out]))
elif args.dataset_out in {'SVHN'}:
    ood_data = ood_dataset_dict[args.dataset_out](paths_config.location_dict['SVHN'], split='train', download=True,
                                                  transform=trn.Compose([pre_transform_dict[args.dataset_out],
                                                                     train_transform_out]))
else:
    raise NotImplementedError(f'Dataset not available as surrogate OOD {args.dataset_out}.')

if args.out_data_number_limit > 0:
    data_step = len(ood_data) // args.out_data_number_limit
    #subset_indices = range(0, data_step*args.out_data_number_limit, data_step)
    subset_indices = range(0, args.out_data_number_limit)
    ood_data = torch.utils.data.Subset(ood_data, subset_indices)

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.in_batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.out_batch_size, shuffle=train_shuffle_dict[args.dataset_out],
    num_workers=args.num_workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data_in,
    batch_size=args.test_bs, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)

# Create model
num_outputs = specs.get_num_outputs(args.method, args.dataset_in)
if args.architecture_base == 'wrn':
    net = WideResNet(args.layers, num_outputs, args.widen_factor, dropRate=args.droprate)
else:
    raise NotImplementedError(f'Base architecture {args.architecture_base}')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
if args.ngpu > 0:
    net.cuda()

exp_str = f'{args.dataset_in}_{args.method}_{args.dataset_out}'
for k, v in getattr(default_args, args.method).items():
    arg_v = getattr(args, k)
    if arg_v != v or k in args.show_args:
        exp_str += f'_{k}={arg_v}'

start_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

optimizer = torch.optim.SGD(
    net.parameters(), args_dict['learning_rate'], momentum=args_dict['momentum'],
    weight_decay=args_dict['weight_decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////
def train():
    net.train()  # enter train mode
    losses_avg_dict = {}
    if args.dataset_out == 'TINY':
        pass
        # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
        #train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))  # should commute with iter
    train_iter_out = iter(train_loader_out)
    for batch_idx, batch_loaded_in in enumerate(train_loader_in):  # currently also loaded for SepD
        try:
            batch_loaded_out = next(train_iter_out)  # raises exception if train_loader_out is shorter.
            # currently also loaded for Plain
        except StopIteration:
            train_iter_out = iter(train_loader_out)
            batch_loaded_out = next(train_iter_out)
        if args.method in {'OE', 'BGC', 'SharedCD', 'SepD'} or args.pass_all:
            data = torch.cat((batch_loaded_in[0], batch_loaded_out[0]), dim=0)
            # all inputs are passed in one batch to avoid BatchNorm and DropOut inconsistencies
            # print(torch.tensor(specs.cifar_std).shape)
            # print(batch_loaded_out[0].shape)
            # torchvision.utils.save_image(batch_loaded_out[0]
            #                              * torch.tensor(specs.cifar_std).unsqueeze(0).unsqueeze(2).unsqueeze(2)
            #                              + torch.tensor(specs.cifar_mean).unsqueeze(0).unsqueeze(2).unsqueeze(2),
            #                              os.path.join(args.save, f'{exp_str}_{start_str}_out_batch_{epoch}.png'))
            # raise Exception('img save run')
        elif args.method in {'Plain', 'InMultiClassBin', 'SharedInMultiClassBin'}:
            data = batch_loaded_in[0]
        else:
            raise NotImplementedError(f'Method {args.method} with pass_all set to {args.pass_all}.')
        labels_in = batch_loaded_in[1]
        if args.ngpu > 0:
            data, labels_in = data.cuda(), labels_in.cuda()
        bs_in = len(batch_loaded_in[0])
        # bs_out = len(batch_loaded_out[0])

        # forward
        x = net(data)

        # backward
        scheduler.step()  # same as [OE]
        optimizer.zero_grad()

        if args.method in {'OE'}:
            loss_parts = {
                'CE_to_labels_in': F.cross_entropy(x[:bs_in], labels_in, reduction='mean'),
                'CE_to_uni_out': -(x[bs_in:].mean(1) - torch.logsumexp(x[bs_in:], dim=1)).mean()
            }
            loss = loss_parts['CE_to_labels_in'] + args.lamb * loss_parts['CE_to_uni_out']
        elif args.method in {'BGC'}:
            loss_parts = {
                'CE_to_labels_in': F.cross_entropy(x[:bs_in], labels_in, reduction='mean'),
                'CE_to_bgc_out': F.cross_entropy(x[bs_in:], torch.full((len(x[bs_in:]),), num_in_classes,
                                                                       dtype=torch.long, device=x.device),
                                                 reduction='mean')
            }
            loss = loss_parts['CE_to_labels_in'] + args.lamb * loss_parts['CE_to_bgc_out']
        elif args.method in {'SharedCD'}:
            loss_parts = {
                'CE_to_labels_in': F.cross_entropy(x[:bs_in][:, :num_in_classes], labels_in, reduction='mean'),
                'CE_disc_in': -F.logsigmoid(x[:bs_in][:, num_in_classes]).mean(),
                'CE_disc_out': -F.logsigmoid(-x[bs_in:][:, num_in_classes]).mean(),
            }
            loss = loss_parts['CE_to_labels_in'] + loss_parts['CE_disc_in'] + args.lamb * loss_parts['CE_disc_out']
        elif args.method in {'Plain'}:
            if args.pass_all:
                x = x[:bs_in]
            else:
                x = x
            loss_parts = dict([])
            loss = F.cross_entropy(x, labels_in, reduction='mean')
        elif args.method in {'InMultiClassBin'}:
            if args.pass_all:
                x = x[:bs_in]
            else:
                x = x
            loss_parts = dict([])
            classwise_label = 2*torch.nn.functional.one_hot(labels_in, num_in_classes) - 1 #+1 for correct class, -1 for others
            loss = -F.logsigmoid(x*classwise_label).mean()
        elif args.method in {'SharedInMultiClassBin'}:
            if args.pass_all:
                x = x[:bs_in]
            else:
                x = x
            classwise_label = 2*torch.nn.functional.one_hot(labels_in, num_in_classes) - 1 #+1 for correct class, -1 for others
            loss_parts = {
                'CE_to_labels': F.cross_entropy(x[:,:num_in_classes], labels_in, reduction='mean'),
                'CE_binary': -F.logsigmoid(x[:,num_in_classes:]*classwise_label).mean()
            }
            loss = loss_parts['CE_to_labels'] + args.lamb * loss_parts['CE_binary']
        elif args.method in {'SepD'}:
            loss_parts = {
                'CE_disc_in': -F.logsigmoid(x[:bs_in]).mean(),
                'CE_disc_out': -F.logsigmoid(-x[bs_in:]).mean(),
            }
            loss = loss_parts['CE_disc_in'] + args.lamb * loss_parts['CE_disc_out']
        else:
            raise NotImplementedError(f'Method {args.method}.')
        loss.backward()
        optimizer.step()

        # exponential moving average
        ema = 0.2
        loss_parts['loss'] = loss
        for k, v in loss_parts.items():
            if k not in losses_avg_dict.keys():
                losses_avg_dict[k] = float(v)
            else:
                losses_avg_dict[k] = losses_avg_dict[k] * (1-ema) + float(v) * ema
    return losses_avg_dict


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)

            # accuracy
            if args.method in {'Plain', 'OE'}:
                loss = F.cross_entropy(output, target)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
            elif args.method in {'BGC', 'SharedCD'}:
                loss = F.cross_entropy(output[:, :num_in_classes], target)
                pred = output[:, :num_in_classes].data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
            elif args.method in {'SepD'}:
                loss = -F.logsigmoid(output).mean()
                pred = output.data >= 0
                correct += pred.sum().item()
            elif args.method in {'InMultiClassBin'}:
                classwise_label = 2 * torch.nn.functional.one_hot(target, num_in_classes) - 1  # +1 for correct class, -1 for others
                loss = -F.logsigmoid(output * classwise_label).mean()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
            elif args.method in {'SharedInMultiClassBin'}:
                classwise_label = 2 * torch.nn.functional.one_hot(target, num_in_classes) - 1  # +1 for correct class, -1 for others
                loss = -F.logsigmoid(output[:,num_in_classes:] * classwise_label).mean()
                pred = output[:,num_in_classes:].data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
            else:
                raise NotImplementedError(f'Method {args.method}.')
            # test loss average
            loss_avg += float(loss.data)

    args_dict['test_loss'] = loss_avg / len(test_loader)
    args_dict['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(args_dict)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)
if not os.path.isdir(args.save):
    raise Exception(f'{args.save} is not a dir')
# using time in saving locations should prevent files being overwritten, but we make sure
if any(File.startswith(f'{exp_str}_{start_str}') for File in os.listdir(args.save)):
    raise FileExistsError(f'There is a training run with this name prefix {exp_str}_{start_str} already.')

# Main loop
print('Beginning Training\n')
for epoch in range(0, args.epochs):
    args_dict['epoch'] = epoch

    begin_epoch = time.time()

    train_losses_avg_dict = train()
    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, f'{exp_str}_{start_str}_epoch_{epoch}.pt'))
    info_dict = {k: v for k, v in args_dict.items()}
    info_dict['epoch'] = epoch
    info_dict['model_name'] = exp_str
    info_dict['num_outputs'] = num_outputs
    with open(os.path.join(args.save, f'{exp_str}_{start_str}_epoch_{epoch}.json'), 'w+') as jf:
        json.dump(info_dict, jf)
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, f'{exp_str}_{start_str}_epoch_{epoch-1}.pt')
    prev_json = os.path.join(args.save, f'{exp_str}_{start_str}_epoch_{epoch-1}.json')
    if os.path.exists(prev_path):
        os.remove(prev_path)
        os.remove(prev_json)

    # Show results
    training_results_file_dest = os.path.join(args.save, f'{exp_str}_{start_str}_training_metrics.csv')
    metrics_names = ['epoch', 'time(s)', 'test_loss', 'test_acc(%)'] + \
                    [f'{loss_name}' for loss_name in train_losses_avg_dict.keys()]
    if not os.path.isfile(training_results_file_dest):
        with open(training_results_file_dest, 'w') as f:
            f.write(f'{",".join(metrics_names)}\n')
    with open(training_results_file_dest, 'a') as f:
        f.write(f'{epoch:03d},'
                f'{int(time.time() - begin_epoch):05d},'
                f'{args_dict["test_loss"]:.3f},'
                f'{100. * args_dict["test_accuracy"]:.3f},'
                f'{",".join(f"{loss_avg:.3f}" for _, loss_avg in train_losses_avg_dict.items())}'
                '\n')
    print(f'Epoch {epoch:03d}'
          f' | Time {int(time.time() - begin_epoch):05d}'
          f' | Test Loss {args_dict["test_loss"]:.3f}'
          f' | Test Accuracy {100. * args_dict["test_accuracy"]:.3f}'
          f' | Train Loss {train_losses_avg_dict["loss"]:.2f}')

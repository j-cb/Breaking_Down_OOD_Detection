from models.wrn import WideResNet
import os
import json
import torch
import re


def load_model(location, model_info, device):
    if model_info['architecture_base'] == 'wrn':
        net = WideResNet(model_info['layers'], model_info['num_outputs'],
                         model_info['widen_factor'], model_info['droprate'])
    else:
        raise NotImplementedError(f'Architecture base {model_info.architecture_base}.')
    net.to(device)
    loaded_weights = torch.load(location, map_location='cpu')
    if net.fc.out_features == 1 and loaded_weights['fc.weight'].shape[0] > 1:
        loaded_weights['fc.weight'] = loaded_weights['fc.weight'][:1]
        loaded_weights['fc.bias'] = loaded_weights['fc.bias'][:1]
        net.load_state_dict(loaded_weights)
    else:
        net.load_state_dict(loaded_weights)
    return net


def load_saved_info(location, not_existing_ok=True):
    if location[-3:] != '.pt':
        raise ValueError(f'The location {location} does not end in \".pt\".')
    json_location = location[:-3] + '.json'
    if os.path.exists(json_location):
        with open(json_location, 'r') as jf:
            saved_info = json.load(jf)
        return saved_info
    else:
        if not_existing_ok:
            return dict([])
        else:
            raise FileNotFoundError(f'No json file {json_location}')


def location_to_generic_name(location):
    generic_info = dict([])
    filename = location.split(r'/')[-1]
    generic_info['model_name'] = filename
    date_match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', filename)
    if date_match is None:
        generic_info['date'] = 'unknown_date'
    else:
        generic_info['date'] = date_match.group(0)
    methods_str_dict = {
        'plain': 'Plain',
        'baseline': 'Plain',
        'sharedinmulticlassbin': 'SharedInMultiClassBin',
        'inmulticlassbin': 'InMultiClassBin',
        'bgc': 'BGC',
        'extra': 'BGC',
        'shared': 'SharedCD',
        'sep': 'SepD',
        'oe': 'OE',
    }
    for k, v in methods_str_dict.items():
        method_match = re.search(k, filename.lower())
        if method_match is not None:
            generic_info['method'] = v
            break
    else:
        generic_info['method'] = 'unknown_method'
    if re.search('cifar100', filename.lower()) is not None:  # this needs to be reworked if there are more datasets
        generic_info['dataset_in'] = 'CIFAR100'
    else:
        generic_info['dataset_in'] = 'CIFAR10'
    if re.search('openimages', filename.lower()) is not None:  # this needs to be reworked if there are more datasets
        generic_info['dataset_out'] = 'OpenImages'
    else:
        generic_info['dataset_out'] = 'TINY'
    return generic_info

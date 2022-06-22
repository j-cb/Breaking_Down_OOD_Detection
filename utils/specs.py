import numpy as np
import scipy


def get_num_in_classes(dataset_in):
    if dataset_in == 'CIFAR10':
        num_in_classes = 10
    elif dataset_in == 'CIFAR100':
        num_in_classes = 100
    else:
        raise NotImplementedError(f'Number of classes of {dataset_in} dataset not known.')
    return num_in_classes


def get_num_outputs(method, dataset_in):
    num_in_classes = get_num_in_classes(dataset_in)
    num_outputs_dict = {
        'Plain': num_in_classes,
        'OE': num_in_classes,
        'BGC': num_in_classes + 1,
        'SharedCD': num_in_classes + 1,
        'SepD': 1,
        'InMultiClassBin': num_in_classes,
        'SharedInMultiClassBin': 2*num_in_classes,
    }
    return num_outputs_dict[method]


# mean and standard deviation of channels of CIFAR-10 images.py, see [OE]
cifar_mean = [x / 255 for x in [125.3, 123.0, 113.9]]
cifar_std = [x / 255 for x in [63.0, 62.1, 66.7]]

potential_OOD_names = ['SVHN', 'LSUN_CR', 'Uniform', 'Smooth', 'CIFAR10', 'CIFAR100', 'TINY', 'OpenImages', 'CelebA']


def get_unseen_OOD_names(dataset_in, dataset_out):
    ood_names = [d for d in potential_OOD_names if d not in {dataset_in, dataset_out}]
    if dataset_in == 'CIFAR100':
        ood_names = [d if d != 'CelebA' else '-' for d in ood_names]
    return ood_names


# noinspection PyUnresolvedReferences
def get_preds_and_scores(method, logits, combi_logits=None, num_in_classes=None):
    if method in {'Plain', 'OE'}:
        preds_and_scores = {
            'confidences': scipy.special.softmax(logits, axis=1).max(axis=1),
            'class_probabilities': scipy.special.softmax(logits, axis=1),
            'predicted_classes': logits.argmax(axis=1)
        }
    elif method in {'BGC'}:
        preds_and_scores = {
            'predicted_overall_classes': logits.argmax(axis=1),
            'predicted_in_classes': logits[:, :num_in_classes].argmax(axis=1),
            'confidences_in_logits': scipy.special.softmax(logits[:, :num_in_classes], axis=1).max(axis=1),
            's2': scipy.special.softmax(logits, axis=1)[:, :num_in_classes].max(axis=1),
            'confidences_incl_bgc': scipy.special.softmax(logits, axis=1).max(axis=1),
            's1': 1 - scipy.special.softmax(logits, axis=1)[:, num_in_classes],
        }
        preds_and_scores['s3'] = preds_and_scores['s2'] + 1 / num_in_classes * (1 - preds_and_scores['s1'])
    elif method in {'SharedCD'}:
        preds_and_scores = {
            'predicted_classes': logits[:, :num_in_classes].argmax(axis=1),
            'class_probabilities_in': scipy.special.softmax(logits[:, :num_in_classes], axis=1),
            'confidences_classifier': scipy.special.softmax(logits[:, :num_in_classes], axis=1).max(axis=1),
            's1': scipy.special.expit(logits[:, num_in_classes]),
        }
        preds_and_scores['s2'] = preds_and_scores['confidences_classifier'] * preds_and_scores['s1']
        preds_and_scores['s3'] = preds_and_scores['s2'] + 1 / num_in_classes * (1 - preds_and_scores['s1'])
        # power = doing the combination with the classifier multiple times
        preds_and_scores['s2_power'] = {r: preds_and_scores['s1']**r * preds_and_scores['confidences_classifier']
                                         for r in np.linspace(0, 1, 100)}
        preds_and_scores['s2_power'].update({r: preds_and_scores['s1'] * preds_and_scores['confidences_classifier']**(1/r)
                                        for r in np.linspace(1, 2, 100)})
        preds_and_scores['s3_power'] = {r: (preds_and_scores['s1'])**r
                                          * (preds_and_scores['confidences_classifier'] - 1 / num_in_classes)
                                        for r in np.linspace(0, 1, 100)}  # removed constant 1 / num_in_classes
        preds_and_scores['s3_power'].update({r: (preds_and_scores['s1'])
                                            * (preds_and_scores['confidences_classifier'] - 1 / num_in_classes)**(1/r)
                                            for r in np.linspace(1, 4, 100)})
        # lambda with p_\lambda(i|x) = p_1(i|x) / (\lambda + (1-\lambda)\cdot p_1(i|x))
        preds_and_scores['s1_lambda'] = {r: preds_and_scores['s1']/(r + (1-r) * preds_and_scores['s1'])
                                         for r in np.linspace(0, 20, 100)}
        preds_and_scores['s2_lambda'] = {r: preds_and_scores['confidences_classifier'] * preds_and_scores['s1_lambda'][r]
                                         for r in preds_and_scores['s1_lambda'].keys()}
        preds_and_scores['s3_lambda'] = {r: preds_and_scores['s2_lambda'][r]
                                         + 1 / num_in_classes * (1 - preds_and_scores['s1_lambda'][r])
                                         for r in preds_and_scores['s1_lambda'].keys()}
    elif method in {'SepD'}:
        preds_and_scores = {
            's1': scipy.special.expit(logits[:, 0]),
        }
        if combi_logits is not None:
            preds_and_scores['confidences_classifier'] = scipy.special.softmax(
                combi_logits, axis=1).max(axis=1)
            preds_and_scores['s2'] = preds_and_scores['confidences_classifier'] * preds_and_scores['s1']
            preds_and_scores['s3'] = preds_and_scores['s2'] + 1 / num_in_classes * (1 - preds_and_scores['s1'])
    elif method in {'InMultiClassBin'}:
        preds_and_scores = {
            'confidences': scipy.special.expit(logits).max(axis=1),
            'class_probabilities': scipy.special.expit(logits)/scipy.special.expit(logits).sum(axis=1, keepdims=True),
            'predicted_classes': logits.argmax(axis=1)
        }
    elif method in {'SharedInMultiClassBin'}:
        logits = logits[:,num_in_classes:]
        preds_and_scores = {
            'confidences': scipy.special.expit(logits).max(axis=1),
            'class_probabilities': scipy.special.expit(logits)/scipy.special.expit(logits).sum(axis=1, keepdims=True),
            'predicted_classes': logits.argmax(axis=1)
        }
    else:
        raise NotImplementedError(f'Scoring functions for the {method} method not known.')
    return preds_and_scores

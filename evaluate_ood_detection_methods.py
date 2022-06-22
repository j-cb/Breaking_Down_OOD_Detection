import argparse
import datetime
import json
import os

import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import rankdata

import utils.dataloading as dataloading
import utils.default_args as default_args
import utils.specs as specs
import utils.metrics as metrics
from utils.metrics import auroc, auprc5r, fpr_at_tpr
from utils.model_loading import load_model, load_saved_info, location_to_generic_name


def get_metrics_from_scores(in_scores, out_scores):
    if out_scores is None:
        metrics_values = None
    else:
        metrics_values = {'AUROC': auroc(in_scores, out_scores),
                          'FPR95': fpr_at_tpr(in_scores, out_scores, 0.95),
                          'AUPR': auroc(in_scores, out_scores), # CHANGE BACK!!!! NaNauprc5r(in_scores, out_scores)
                          }
    return metrics_values


def evaluate_model(model, method, dataset_in, eval_OOD_names, device,
                   max_samples=30080, num_workers=4, test_bs=200,
                   save_folder_outputs=None, load_folder_outputs=None, combi_logits_folder=None):
    dataloader_kwargs = {'num_workers': num_workers, 'pin_memory': True}
    test_loader_in = dataloading.datasets_dict[dataset_in](train=False, batch_size=test_bs, augmentation=dict([]),
                                                           dataloader_kwargs=dataloader_kwargs)
    test_loader_out_dict = dataloading.get_test_out_loaders(test_bs, dataloader_kwargs,
                                                            {name: dict([]) for name in eval_OOD_names})
    preds_and_scores_dict = {}

    model.eval()
    num_in_classes = specs.get_num_in_classes(dataset_in)
    num_outputs = specs.get_num_outputs(method, dataset_in)

    # get predictions on the in-distribution
    if load_folder_outputs is None:
        full_logits_in = np.zeros((0, num_outputs))
        labels_in = np.zeros(0)
        for i, data in enumerate(test_loader_in, 0):
            samples, labels = [x.to(device) for x in data]
            inputs = (samples - torch.tensor(specs.cifar_mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)) / \
                     torch.tensor(specs.cifar_std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
            outputs = model(inputs)
            labels_in = np.append(labels_in, labels.data.cpu().numpy(), axis=0)
            full_logits_in = np.append(full_logits_in, outputs.data.cpu().numpy(), axis=0)
            if len(full_logits_in) >= max_samples:
                full_logits_in = full_logits_in[:max_samples]
                break
        if save_folder_outputs:
            np.save(os.path.join(save_folder_outputs, 'full_logits_in.npy'), full_logits_in)
            np.save(os.path.join(save_folder_outputs, 'labels_in.npy'), labels_in)
    else:
        full_logits_in = np.load(os.path.join(load_folder_outputs, f'full_logits_in.npy'))
        labels_in = np.load(os.path.join(load_folder_outputs, f'labels_in.npy'))
    if combi_logits_folder:
        combi_logits_in = np.load(os.path.join(combi_logits_folder, f'full_logits_in.npy'))
    else:
        combi_logits_in = None
    preds_and_scores_dict['in'] = specs.get_preds_and_scores(method, full_logits_in, combi_logits_in, num_in_classes)


    if 'in' in eval_OOD_names:
        raise ValueError('OOD name cannot be \"in\"')

    # get predictions on the out-distributions
    for s in eval_OOD_names:
        if load_folder_outputs is None:
            test_loader_out = test_loader_out_dict[s]
            if test_loader_out is None:
                preds_and_scores_dict[s] = None
                continue  # no saving done
            full_logits_out = np.zeros((0, num_outputs))
            for i, data in enumerate(test_loader_out, 0):
                samples, labels = [x.to(device) for x in data]
                inputs = (samples - torch.tensor(specs.cifar_mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)) / \
                         torch.tensor(specs.cifar_std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
                outputs = model(inputs)
                full_logits_out = np.append(full_logits_out, outputs.data.cpu().numpy(), axis=0)
                if len(full_logits_out) >= max_samples:
                    full_logits_out = full_logits_out[:max_samples]
                    break
            if save_folder_outputs:
                np.save(os.path.join(save_folder_outputs, f'full_logits_out_{s}.npy'), full_logits_out)
        else:
            print(f'loading previous eval outputs from {os.path.join(load_folder_outputs, f"full_logits_out_{s}.npy")}')
            full_logits_out_path = os.path.join(load_folder_outputs, f'full_logits_out_{s}.npy')
            if os.path.exists(full_logits_out_path):
                full_logits_out = np.load(full_logits_out_path)
            else:
                preds_and_scores_dict[s] = None
                continue
        if combi_logits_folder:
            combi_logits = np.load(os.path.join(combi_logits_folder, f'full_logits_out_{s}.npy'))
        else:
            combi_logits = None
        preds_and_scores_dict[s] = specs.get_preds_and_scores(method, full_logits_out, combi_logits, num_in_classes)

    return preds_and_scores_dict, labels_in, test_loader_in, test_loader_out_dict


def calc_metrics(preds_and_scores_dict, method, eval_OOD_names, labels_in):
    results = {}
    if method in {'SharedCD'}:
        results['classi_acc'] = np.equal(preds_and_scores_dict['in']['predicted_classes'], labels_in).mean()
        results['classi_ece'] = metrics.expected_calibration_error(
            preds_and_scores_dict['in']['predicted_classes'],
            preds_and_scores_dict['in']['confidences_classifier'],
            labels_in
        )
        results['classi_CE'] = metrics.cross_entropy(preds_and_scores_dict['in']['class_probabilities_in'], labels_in)
        possible_scores = ['s1', 'confidences_classifier', 's2', 's3']
        possible_calibration_metrics = ['classi_ece', 'classi_CE']
        possible_graphs = ['s2_power', 's3_power', 's1_lambda', 's2_lambda', 's3_lambda']
    elif method in {'BGC'}:
        results['classi_acc'] = np.equal(preds_and_scores_dict['in']['predicted_in_classes'], labels_in).mean()
        results['all_classes_ece'] = metrics.expected_calibration_error(
            preds_and_scores_dict['in']['predicted_overall_classes'],
            preds_and_scores_dict['in']['confidences_incl_bgc'],
            labels_in
        )
        results['in_classes_ece'] = metrics.expected_calibration_error(
            preds_and_scores_dict['in']['predicted_in_classes'],
            preds_and_scores_dict['in']['confidences_in_logits'],
            labels_in
        )
        possible_scores = ['s1', 's2', 's3']
        possible_calibration_metrics = ['all_classes_ece', 'in_classes_ece']
        possible_graphs = []
    elif method in {'Plain', 'OE', 'InMultiClassBin', 'SharedInMultiClassBin'}:
        results['classi_acc'] = np.equal(preds_and_scores_dict['in']['predicted_classes'], labels_in).mean()
        results['classi_ece'] = metrics.expected_calibration_error(
            preds_and_scores_dict['in']['predicted_classes'],
            preds_and_scores_dict['in']['confidences'],
            labels_in
        )
        possible_scores = ['confidences']
        possible_calibration_metrics = ['classi_ece']
        possible_graphs = []
    elif method in {'SepD'}:
        possible_scores = ['s1']
        possible_calibration_metrics = []
        possible_graphs = []
    else:
        raise NotImplementedError(f'Method {method} has no scores')
    for score in possible_scores:
        results[score] = {s: get_metrics_from_scores(preds_and_scores_dict['in'][score],#.get(score),
                                                     preds_and_scores_dict[s][score])#.get(score))
                             if preds_and_scores_dict[s] is not None
                             else None
                          for s in eval_OOD_names}
    for score in possible_graphs:
        results[score] = {s: {r: get_metrics_from_scores(preds_and_scores_dict['in'][score][r],
                                                         preds_and_scores_dict[s][score][r])
                              for r in preds_and_scores_dict['in'][score].keys()}
                              if preds_and_scores_dict[s] is not None
                              else None
                          for s in eval_OOD_names}
    if {'s1','s2'} <= set(possible_scores):
        k = 10
        results['s2<s1'], results['s2>s1'] = {}, {}
        for s in ['in'] + eval_OOD_names:
            if preds_and_scores_dict[s] is None:
                results['s2<s1'][s] = None
                results['s2>s1'][s] = None
                continue
            rank_s1_minus_s2 = rankdata(preds_and_scores_dict[s]['s1']) - rankdata(preds_and_scores_dict[s]['s2'])
            largest_differences = np.argsort(rank_s1_minus_s2)
            results['s2<s1'][s] = (largest_differences[:k], rank_s1_minus_s2[largest_differences[:k]])
            results['s2>s1'][s] = (largest_differences[-k:], rank_s1_minus_s2[largest_differences[-k:]])
    if 's3' in possible_scores or 'confidences' in possible_scores:
        if 's3' in possible_scores:
            score = 's3'
        else:
            score = 'confidences'
        k = 60
        results['highest_s3'], results['lowest_s3'] = {}, {}
        for s in ['in'] + eval_OOD_names:
            if preds_and_scores_dict[s] is None:
                results['highest_s3'][s] = None
                results['lowest_s3'][s] = None
                continue
            highest_s3 = np.argsort(preds_and_scores_dict[s][score])
            results['lowest_s3'][s] = (highest_s3[:k], preds_and_scores_dict[s][score][highest_s3[:k]])
            results['highest_s3'][s] = (highest_s3[-k:], preds_and_scores_dict[s][score][highest_s3[-k:]])
    results['possible_scores'] = possible_scores
    results['possible_calibration_metrics'] = possible_calibration_metrics
    results['possible_graphs'] = possible_graphs
    return results


def latex_len(s, n):
    if len(s) < 5:
        return '\\phantom{' + (n - len(s)) * '0' + '}' + s
    else:
        return s


def results_row_string(model_name, results, eval_OOD_names, unseen_OOD_names, score_name, metrics_name,
                       include_acc=True):
    entries_perc = []
    for d in eval_OOD_names:
        if results[score_name][d] is None:
            entries_perc.append(f'?')
        else:
            entries_perc.append(f'{100 * results[score_name][d][metrics_name]:.2f}')
    rs = ''
    rs += score_name + '--' + model_name.replace('_', '-')
    if include_acc:
        rs += '\n &  ' + f'{100 * results["classi_acc"]:.2f}'
    else:
        rs += '\n &  ' + f' '

    mean_value = np.mean([results[score_name][d][metrics_name]
                          for d in unseen_OOD_names if results[score_name][d] is not None])
    rs += '\n &  ' + latex_len(f'{100 * mean_value:.2f}', 5) + '  '
    rs += f'  %{metrics_name}'
    for s in entries_perc:
        rs += '\n &  ' + latex_len(s, 5) + '  '
    rs += r' \\'
    return rs


dset_titles = {
    'TINY': '80M'
}


def column_titles_string(OOD_names):
    column_titles = []
    for dset_name in OOD_names:
        column_titles.append(f'{dset_titles.get(dset_name, dset_name)}')
    ts = r'Model & Acc. & Mean '
    for t in column_titles:
        ts += f' &  {t} '  # {latex_len(t, 10)}'
    ts += r' \\'
    return ts


def complete_info(model_info, defaults_dict=None, overwrites_dict=None, generic_info=None):
    if defaults_dict is None:
        defaults_dict = dict([])
    if generic_info is not None:
        defaults_dict.update(generic_info)
    if model_info is not None:
        defaults_dict.update(model_info)
    if overwrites_dict is not None:
        defaults_dict.update(overwrites_dict)
    # if 'num_outputs' not in defaults_dict.keys():
    defaults_dict['num_outputs'] = specs.get_num_outputs(defaults_dict['method'], defaults_dict['dataset_in'])
    return defaults_dict


def evaluate_saves(saves_json, device, max_samples=30080, num_workers=4, test_bs=200, save_folder=None,
                   load_outputs=None, unseen_OOD_arg=None, save_images=False):
    eval_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(saves_json, 'r') as jf:
        saves_dict = json.load(jf)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for location, model_info in saves_dict.items():
        if location[:2] == '__':
            print(f'Skipping{location}.')
            continue
        else:
            print(f'Evaluating {location}.')
        generic_info = location_to_generic_name(location)
        if not model_info:
            model_info = load_saved_info(location, not_existing_ok=True)
        if 'method' not in model_info.keys():
            model_info['method'] = generic_info['method']
        full_model_info = complete_info(model_info, defaults_dict=getattr(default_args, model_info['method']),
                                        generic_info=generic_info)
        details_save_folder = os.path.join(save_folder, f'{full_model_info["model_name"]}_E{eval_time}')
        if load_outputs == 'last':
            load_folder_outputs = sorted([dirpath for dirpath in
                        [os.path.join(save_folder,path) for path in os.listdir(save_folder)
                        if path[:len(full_model_info["model_name"])] == full_model_info["model_name"]]
                    if os.path.isdir(dirpath) and 'full_logits_in.npy' in os.listdir(dirpath)])[-1]
        else:
            load_folder_outputs = None
        if not os.path.exists(details_save_folder):
            os.makedirs(details_save_folder)
        model = load_model(location, full_model_info, device)
        if full_model_info['dataset_out'] == 'OpenImages32':
            train_dataset_out_name = 'OpenImages'
        else:
            train_dataset_out_name = full_model_info['dataset_out']
        if unseen_OOD_arg is None or 'all' in unseen_OOD_arg:
            unseen_OOD_names = specs.get_unseen_OOD_names(full_model_info['dataset_in'], train_dataset_out_name)
        else:
            unseen_OOD_names = unseen_OOD_arg
        eval_OOD_names = unseen_OOD_names + [train_dataset_out_name]
        preds_and_scores_dict, labels_in, test_loader_in, test_loader_out_dict = evaluate_model(model, method=full_model_info['method'],
                                                          dataset_in=full_model_info['dataset_in'],
                                                          eval_OOD_names=eval_OOD_names,
                                                          device=device,
                                                          max_samples=max_samples,
                                                          num_workers=num_workers,
                                                          test_bs=test_bs,
                                                          save_folder_outputs=details_save_folder,
                                                          load_folder_outputs=load_folder_outputs,
                                                          combi_logits_folder=full_model_info.get('combi_logits_folder')
                                                          )
        results = calc_metrics(preds_and_scores_dict, full_model_info['method'], eval_OOD_names, labels_in)
        latex_table_save_file = os.path.join(save_folder, f'{full_model_info["model_name"]}_E{eval_time}.txt')
        with open(latex_table_save_file, 'w') as f:
            f.write(f'\n------{location}------{full_model_info["model_name"]}------------\n')
            for m in ['AUROC', 'FPR95', 'AUPR']:
                f.write(f'\n----------------{m}---------------\n')
                f.write(f'{column_titles_string(eval_OOD_names)}\n' + r'\midrule' + '\n')
                for score_name in results['possible_scores']:
                    row = results_row_string(full_model_info["model_name"], results, eval_OOD_names,
                                             unseen_OOD_names, score_name, m,
                                             include_acc=('classi_acc' in results.keys()))
                    f.write(f'{row}\n')
            for cali in results['possible_calibration_metrics']:
                f.write(f'\n\n{cali}:\n{results[cali][0]}\n')
                f.write(f'Bin details:\n{results[cali][1]}')
        if not save_images:
            continue
        for graph_name in results['possible_graphs']:
            for m in ['AUROC']:
                # p = figure(title="score_name", x_axis_label='power', y_axis_label=m, output_backend="svg")
                for ood in results[graph_name].keys():
                    if results[graph_name][ood] is None:
                        continue
                    else:
                        x_values = np.array(list(results[graph_name][ood].keys()))
                        y_values = np.array([results[graph_name][ood][x][m] for x in x_values])
                        ymax = max(y_values)
                        xpos = np.argmax(y_values)
                        xmax = x_values[xpos]
                        max_str = f'({xmax:.3f}|{ymax:.3f})'
                        plt.plot(x_values, y_values, label=f'{ood: >12} {max_str}', linewidth=2)
                plt.legend(loc="lower right")
                plt.title(f'{graph_name}_{m}')
                plt.savefig(os.path.join(details_save_folder, f'{graph_name}_{m}.png'))
                plt.close()
        if 's2>s1' in results.keys():
            score_differences_file = os.path.join(details_save_folder, f'score_differences_E{eval_time}.txt')
            k = 10
            fig, ax = plt.subplots(4 * len(results['s2>s1'].keys()), k+1, dpi=100,
                                   figsize=(k+1, 4 * len(results['s2>s1'].keys())))
            num_image_rows = 0
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
            with open(score_differences_file, 'w') as f:
                f.write(f'\n------{location}------{full_model_info["model_name"]}------------\n')
                for s in results['s2>s1'].keys():
                    if results['s2>s1'][s] is None:
                        continue
                    f.write(f'\n\n{s}:')
                    for diff in {'s2>s1', 's2<s1'}:
                        indices_str, differences_str = '', ''
                        for i in results[diff][s][0]:
                            indices_str += f' {i:5d}'
                        for d in results[diff][s][1]:
                            differences_str += f' {int(d):5d}'
                        f.write(f'\nIndices with largest rank difference {diff}: {indices_str}')
                        f.write(f'\n                         Differences {diff}: {differences_str}')
                        if s == 'in':
                            dset = test_loader_in.dataset
                        else:
                            dset = test_loader_out_dict[s].dataset
                        images = np.array([dset[i][0].detach().cpu().numpy() for i in results[diff][s][0]])
                        for i in range(len(images)):
                            ax[num_image_rows+1, i].imshow(np.transpose(images[i], (1, 2, 0)), norm=normalize)
                            ax[num_image_rows+1, i].axis('off')
                            ax[num_image_rows+0, i].text(0.2, 0.1, results[diff][s][1][i])
                            ax[num_image_rows+0, i].axis('off')
                        ax[num_image_rows+0, i+1].text(0.2, 0.1, diff)
                        ax[num_image_rows+0, i+1].axis('off')
                        ax[num_image_rows+1, i+1].text(0.2, 0.1, s)
                        ax[num_image_rows+1, i+1].axis('off')
                        num_image_rows += 2
            img_save_path = os.path.join(details_save_folder, f'score_differences_E{eval_time}.png')
            plt.savefig(img_save_path, bbox_inches='tight')
            plt.close()
        if 's3' in results.keys():
            s3_extremes_file = os.path.join(details_save_folder, f's3_extremes_E{eval_time}.txt')
            k = 60
            fig, ax = plt.subplots(4 * len(results['highest_s3'].keys()), k+1, dpi=100,
                                   figsize=(k+1, 4 * len(results['highest_s3'].keys())))
            num_image_rows = 0
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
            with open(s3_extremes_file, 'w') as f:
                f.write(f'\n------{location}------{full_model_info["model_name"]}------------\n')
                for s in results['highest_s3'].keys():
                    if results['highest_s3'][s] is None:
                        continue
                    f.write(f'\n\n{s}:')
                    for extr in {'highest_s3', 'lowest_s3'}:
                        indices_str, extremes_str = '', ''
                        for i in results[extr][s][0]:
                            indices_str += f' {i:5d}'
                        for d in results[extr][s][1]:
                            extremes_str += f' {d:.5f}'
                        f.write(f'\nIndices with {extr}: {indices_str}')
                        f.write(f'\n      values {extr}: {extremes_str}')
                        if s == 'in':
                            dset = test_loader_in.dataset
                        else:
                            dset = test_loader_out_dict[s].dataset
                        images = np.array([dset[i][0].detach().cpu().numpy() for i in results[extr][s][0]])
                        for i in range(len(images)):
                            ax[num_image_rows+1, i].imshow(np.transpose(images[i], (1, 2, 0)), norm=normalize)
                            ax[num_image_rows+1, i].axis('off')
                            ax[num_image_rows+0, i].text(0.2, 0.1, f'{results[extr][s][1][i]:.4f}')
                            ax[num_image_rows+0, i].axis('off')
                        ax[num_image_rows+0, i+1].text(0.2, 0.1, extr)
                        ax[num_image_rows+0, i+1].axis('off')
                        ax[num_image_rows+1, i+1].text(0.2, 0.1, s)
                        ax[num_image_rows+1, i+1].axis('off')
                        num_image_rows += 2
            img_save_path = os.path.join(details_save_folder, f's3_extremes_E{eval_time}.png')
            plt.savefig(img_save_path, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate OOD detection models.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required arguments
    parser.add_argument('saves_json', type=str, help='Dictionary with model paths and optional model specs.')
    # This saves_dict should be a json file containing a dict with the save locations to be evaluated as keys
    # and a dict of dataset_in, method as mandatory keys and dataset_out, architecture details and
    # names as optional keys

    parser.add_argument('--dataset_in', type=str, choices=['cifar10', 'cifar100'], default=None,
                        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument('--dataset_out', type=str, choices=['TINY', 'OpenImages'], default='TINY',
                        help='Choose between TINY and OpenImages the out-distribution seen during training.')
    parser.add_argument('--eval_datasets_out', type=str, nargs='*', default='all',
                        help='Choose between TINY and OpenImages the out-distribution seen during training.')
    parser.add_argument('--overwrite_in', action='store_true',
                        help='use the above in-dataset even if one is provided with the model')
    parser.add_argument('--overwrite_out', action='store_true',
                        help='use the above out-dataset even if one is provided with the model')
    parser.add_argument('--combine_with_classifier_eval_json', type=str, help='Dict of classifier evaluations to'
                                                                              ' combine the discriminator with.')

    # WRN Architecture only used if not provided with the model
    parser.add_argument('--architecture_base', '-ab', type=str, default='wrn', help='Choose architecture.',
                        choices=['wrn'])
    parser.add_argument('--layers', '-al', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', '-aw', default=2, type=int, help='widen factor wrn')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    # Acceleration
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use for single gpu')
    parser.add_argument('--num_workers', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--test_bs', type=int, default=200)
    # Saving
    parser.add_argument('--show_args', nargs='*', help='include these hyperparameters is output filenames')
    parser.add_argument('--save_folder', type=str, default='', help='Folder to save outputs.')
    parser.add_argument('--save_images', action='store_true', help='Save samples for high and low scores and score differences.')
    parser.add_argument('--load_outputs', type=str, default='', help='Load previous eval.')

    args = parser.parse_args()

    script_device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(script_device)
    if args.save_folder:
        script_save_folder = args.saves_folder
    else:
        script_save_folder = os.path.join(os.path.dirname(args.saves_json), 'evals')
    evaluate_saves(args.saves_json, script_device, max_samples=30080, num_workers=4,
                   test_bs=200, save_folder=script_save_folder, load_outputs=args.load_outputs, unseen_OOD_arg=args.eval_datasets_out,
                   save_images=args.save_images)

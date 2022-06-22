import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(values_in, values_out):
    if len(values_in)*len(values_out) == 0:
        return np.NAN
    y_true = len(values_in)*[1] + len(values_out)*[0]
    y_score = np.concatenate([np.nan_to_num(values_in, nan=0.0), np.nan_to_num(values_out, nan=0.0)])
    return roc_auc_score(y_true, y_score)


def fpr_at_tpr(values_in, values_out, tpr):
    t = np.quantile(values_in, (1-tpr))
    fpr = (values_out >= t).mean()
    return fpr


def auprc(values_in, values_out):
    y_true = len(values_in)*[1] + len(values_out)*[0]
    y_score = np.concatenate([values_in, values_out])
    return average_precision_score(y_true, y_score)


def auprc5(values_in, values_out):
    if len(values_out) >= len(values_in)//5:
        return auprc(values_in, values_out[:len(values_in) // 5])
    else:
        print(len(values_in[:len(values_out)*5]), len(values_out))
        return auprc(values_in[:len(values_out) * 5], values_out)


def auprc5r(values_in, values_out):
    if len(values_out) >= len(values_in)//5:
        return auprc(-values_out[:len(values_in) // 5], -values_in)
    else:
        return auprc(-values_out, -values_in[:len(values_out) * 5])


def expected_calibration_error(predictions, confidences, labels, n_bins=20):
    ece_str = ''
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = np.equal(predictions, labels)

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.greater(confidences, bin_lower) & np.less_equal(confidences, bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            ece_str += f'{bin_lower.item():.2f} to {bin_upper.item():.2f},' \
                       f'conf-acc = {avg_confidence_in_bin - accuracy_in_bin:.4f}, p(bin) = {prop_in_bin:.4f}\n'
    return ece, ece_str


def cross_entropy(predicted_probabilities, labels):
    prob_for_label_class = predicted_probabilities[np.arange(len(predicted_probabilities)), labels.astype(int)]
    indiv_cross_entropy = -np.log(prob_for_label_class)
    return indiv_cross_entropy.mean(), 'No bins.'

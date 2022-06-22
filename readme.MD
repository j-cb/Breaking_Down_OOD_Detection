Code for the ICML 2022 paper 
# [Breaking Down Out-of-Distribution Detection: Many Methods Based on OOD Training Data Estimate a Combination of the Same Core Quantities](https://arxiv.org/abs/2206.09880)

**Julian Bitterwolf, Alexander Meinke, Max Augustin and Matthias Hein**

**University of Tübingen**

**[https://arxiv.org/abs/2206.09880](https://arxiv.org/abs/2206.09880)**

**Abstract:**
    It is an important problem in trustworthy machine learning to recognize out-of-distribution (OOD) inputs which are inputs unrelated to the in-distribution task. Many out-of-distribution detection methods have been suggested in recent years. The goal of this paper is to recognize common objectives as well as to identify the implicit scoring functions of different OOD detection methods. We focus on the sub-class of methods that use surrogate OOD data during training in order to learn an OOD detection score that generalizes to new unseen out-distributions at test time. We show that binary discrimination between in- and (different) out-distributions is equivalent to several distinct formulations of the OOD detection problem. When trained in a shared fashion with a standard classifier, this binary discriminator reaches an OOD detection performance similar to that of Outlier Exposure. Moreover, we show that the confidence loss which is used by Outlier Exposure has an implicit scoring function which differs in a non-trivial fashion from the theoretically optimal scoring function in the case where training and test out-distribution are the same, which again is similar to the one used when training an Energy-Based OOD detector or when adding a background class. In practice, when trained in exactly the same way, all these methods perform similarly. 

## Training
For training new models, run the `train_ood_aware.py` script with the appropriate arguments.
For example, the shared classifier/binary discriminator for CIFAR-100 can be reproduced by running

`python train_ood_aware.py --dataset_in CIFAR100 --dataset_out OpenImages --gpu 0 --method SharedCD --save ./my_saves` .

## Evaluation
The models from Table 1 in the paper can be evaluated with

`python evaluate_ood_detection_methods.py --gpu 0 trained_models/eval_OpenImages.json` .

To evaluate other models, add them to a `.json` file with their location and relevant hyperparameters, if applicable.

## Setup
The paths in `utils/paths_config.py` need to be set to the locations of your datasets.

If a dataset is not available for evaluation, it will be skipped for the evaluation.

The code was tested with `Python 3.8.5`, `PyTorch 1.6.0` and `torchvision 0.7.0`.

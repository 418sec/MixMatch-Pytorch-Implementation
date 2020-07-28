## MixMatch - Pytorch

This repository is an unofficial implementation of MixMatch with Pytorch. Directory structure of the dataset: [here](https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders?)

**Original Paper**: [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249), by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, Colin Raffel. 

**Original Repo (tensorflow)**: [here](https://github.com/google-research/mixmatch)

## Key mechanisms implemented in this code
1. Augmentation
2. Guess Label
3. Entropy Minimization by Sharpen
4. MixUp
5. Consistensy Loss
6. Exponential Moving Average

The Mean-Teacher model used in this code follows the original implementation, found [here](https://github.com/CuriousAI/mean-teacher)

** This repo applies varied ratio of labels instead of the absolute label amount

## Reference
[MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249), by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, Colin Raffel.

[Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780), by Antti Tarvainen, Harri Valpola

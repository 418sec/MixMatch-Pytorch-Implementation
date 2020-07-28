## MixMatch - Pytorch

This repository is an unofficial implementation of MixMatch with Pytorch.  

**Original Paper**: [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170), by Avital Oliver*, Augustus Odena*, Colin Raffel*, Ekin D. Cubuk, and Ian J. Goodfellow. 

**Original Repo**: [here](https://github.com/brain-research/realistic-ssl-evaluation)

### Key mechanisms implemented in this code
1. Augmentation
2. Guess Label
3. Entropy Minimization by Sharpen
4. MixUp
5. Consistensy Loss
6. Exponential Moving Average

The Mean-Teacher model used in this code follows the original implementation, found [here](https://github.com/CuriousAI/mean-teacher)

** This repo applis varied ratio of labels instead of the absolute labels amount

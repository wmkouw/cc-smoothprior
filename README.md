# Cross-center smoothness prior for hidden Potts models

This repository contains algorithms and experiments for the paper

["A cross-center smoothness prior for variational Bayesian brain tissue segmentation."](https://arxiv.org/abs/1903.04191)

Generalizing machine learning algorithms across medical centers is difficult. Data is often strongly biased towards each center, leading to different mappings from medical image X to segmented image Y.

Instead of designing an adaptive classification model that would attempt to adjust its mapping X &#8594; Y for each center, I inform an unsupervised Bayesian segmentation model with how Y is supposed to look like. Specifically, I fit a smoothness prior on segmentations produced in one medical center and incorporate that as an informative empirical prior in a variational Bayesian image segmentation model. This model will produce segmentations in a target medical center that are as smooth as the segmentations produced in the source medical center.

## Contact

Comments, questions and feedback can be submitted to the [issues tracker](https://github.com/wmkouw/cc-smoothprior/issues).
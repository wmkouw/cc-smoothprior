# Informed priors for hidden Potts-MRFs

In this project, I aim to learn an informative prior for a Bayesian image
segmentation model with a Potts-Markov Random Field as a prior for the labels.

By doing a variational approximation of the Potts prior, we can obtain a form that allows for maximum likelihood estimation of the smoothness parameter \beta. Using an additional data set consisting purely of labeled images, one could obtain an estimate of \beta. Essentially, you are setting the Potts-prior to mimic the smoothness of segmentation in another data set.

Using a similar variational approximation, one could obtain posterior parameter estimates for the whole Bayesian model. Inference consists of doing Expectation-Maximization (EM) on the parameters of the variational distribution.

Further work is in progress.

## Contact

Comments, questions and feedback can be submitted to the [issues tracker](https://github.com/wmkouw/infopriors-hPMRF/issues).
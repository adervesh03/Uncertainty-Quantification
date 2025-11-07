# Uncertainty-Quantification
Automatic vs Bootstrapped Uncertainty Quantification

Our Project will be titled: Automatic versus Bootstrapped Uncertainty Quantification.
That is, we plan to compare SoftBART (A newer version of Bayesian Additive Regression
Trees) with xgBoost and random forests for prediction uncertainty quantification. SoftBART
automatically gives automatic uncertainty quantification due to its Bayesian formalism, while for
xgBoost and random forests, we will need to bootstrap them. After initial simulations, we
hypothesize that SoftBART will have better coverage, particularly for more complicated
data-generating processes, at the expense of greater computational complexity compared to the
other two models.

To evaluate this hypothesis, we plan to perform multiple simulation studies, first, on a
simple model where covariates will enter the model linearly. Second, we plan to extend this to
the case where covariates enter the model nonlinearly with higher-order interactions. Finally, we
will extend to the case of time-series data. In all cases, p<<n due to computational limitations,
and we will explore multiple sample sizes. The exact number of simulations and sample sizes
will be determined later based on the computational requirements of the models. Furthermore,
we plan to investigate the performance of these methods on an iid and a time-series real-world
data set.

glm
===

A lean, mean GLM-modeling machine in Python.

This is loosely based on the GLM facilities in `statsmodels`, but `glm` is
designed to be leaner and simpler. It includes GLM estimation, fast WLS, and a
function for independence-chain Metropolis-Hastings steps with arbitrary GLMs.
The latter is build for easy integration with MCMC routines and has (reasonably)
safe defaults. This package requires only `numpy` and `scipy` beyond Python's
standard library.

The `estimate_glm` script is included with the package. Both read data in a
tabular format (using `numpy.loadtxt`), assuming one file contains the outcome
variable and the other file contains a single predictor per column.  An optional
file of weights can also be included. The `estimate_glm` script will print the
coefficient estimates and, if requested, information and covariance matrices, to
`stdout`.

An upcoming `mcmc_glm` script will use an independence-chain Metropolis-Hastings
algorithm (with a t proposal) to draw for the posterior distribution of a GLM's
coefficients. It will take the same inputs as `estimate_glm` and print the
requested number of MCMC draws to `stdout`.

The Gaussian, binomial, gamma, and Poisson families are currently implemented
with the most commonly used link functions. The negative binomial family will
be added shortly, using an alternating optimization scheme.

All implemented families and links have been validated against R's `glm`
function for a range of settings. They were found to agree with this package to
machine precision using both QR and Cholesky decomposition-based WLS fitting.

This package was originally developed as a small component for Bayesian
inference with complex observation error models in LC/MSMS proteomics, but I
have come see it as a necessary component for much of the Python statistical
community. It will be maintained and community involvement is welcome.

I recommend the [patsy](https://github.com/pydata/patsy) package for building
design matrices from given formulas. You might also find the
[pandas](http://pandas.pydata.org/) package useful for data management; it has
been very carefully built for speed and reliability.


from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import rv_continuous, kstest, wasserstein_distance

type UnivariateDistribution = rv_continuous


class UnivariateDistributionMetric(ABC):
    @abstractmethod
    def evaluate(
        self, distribution: UnivariateDistribution, parameters, data: np.ndarray
    ) -> float:
        pass


class KolmogorovSmirnovTest(UnivariateDistributionMetric):
    def evaluate(
        self, distribution: UnivariateDistribution, parameters, data: np.ndarray
    ) -> float:
        """
        Performs the Kolmogorov-Smirnov test to evaluate goodness of fit.
        Returns the p-value of the test. Higher values indicate better fit.
        """
        # Fit the distribution to the data
        fitted_dist = distribution(*parameters)
        # Perform KS test
        _, p_value = kstest(data, fitted_dist.cdf)
        return p_value  # Already higher is better


class KullbackLeiblerDivergence(UnivariateDistributionMetric):
    def evaluate(
        self, distribution: UnivariateDistribution, parameters, data: np.ndarray
    ) -> float:
        """
        Calculates the negative Kullback-Leibler divergence between the empirical and theoretical distributions.
        Higher values indicate better fit.
        """
        # Fit the distribution to the data
        fitted_dist = distribution(*parameters)

        # Create histogram of empirical data
        hist, bin_edges = np.histogram(data, bins="auto", density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate theoretical probabilities
        theoretical_pdf = fitted_dist.pdf(bin_centers)

        # Calculate KL divergence
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        kl_div_value = np.sum(kl_div(hist + epsilon, theoretical_pdf + epsilon))

        # Return negative KL divergence so higher values are better
        return -kl_div_value


class AIC(UnivariateDistributionMetric):
    def evaluate(
        self, distribution: UnivariateDistribution, parameters, data: np.ndarray
    ) -> float:
        """
        Calculates the negative Akaike Information Criterion.
        Higher values indicate better fit.
        """
        # Fit the distribution to the data
        fitted_dist = distribution(*parameters)

        # Calculate log-likelihood
        log_likelihood = np.sum(fitted_dist.logpdf(data))

        # Number of parameters
        k = len(parameters)

        # Calculate AIC
        aic = 2 * k - 2 * log_likelihood

        # Return negative AIC so higher values are better
        return -aic


class BIC(UnivariateDistributionMetric):
    def evaluate(
        self, distribution: UnivariateDistribution, parameters, data: np.ndarray
    ) -> float:
        """
        Calculates the negative Bayesian Information Criterion.
        Higher values indicate better fit.
        """
        # Fit the distribution to the data
        fitted_dist = distribution(*parameters)

        # Calculate log-likelihood
        log_likelihood = np.sum(fitted_dist.logpdf(data))

        # Number of parameters and sample size
        k = len(parameters)
        n = len(data)

        # Calculate BIC
        bic = np.log(n) * k - 2 * log_likelihood

        # Return negative BIC so higher values are better
        return -bic

class JensenShannonDivergence(UnivariateDistributionMetric):
    def evaluate(
        self,
        distribution: UnivariateDistribution,
        parameters,
        data: np.ndarray
    ) -> float:
        """
        Symmetric and finite divergence even for heavy-tailed distributions.
        Returns negative JS divergence (higher is better).
        """
        fitted_dist = distribution(*parameters)
        hist, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf = fitted_dist.pdf(bin_centers)
        # Ensure valid probability vectors
        eps = 1e-10
        p = hist + eps
        q = pdf + eps
        # SciPy returns sqrt(JS), square to get divergence
        js = jensenshannon(p, q, base=np.e)**2
        return -js


class AlphaDivergence(UnivariateDistributionMetric):
    def __init__(self, alpha: float = 0.5):
        """
        Alpha-divergence family; alpha=1 gives reverse KL, alpha->0 gives forward KL.
        """
        if alpha == 0 or alpha == 1:
            raise ValueError("alpha must be != 0 and != 1 for alpha-divergence")
        self.alpha = alpha

    def evaluate(
        self,
        distribution: UnivariateDistribution,
        parameters,
        data: np.ndarray
    ) -> float:
        fitted_dist = distribution(*parameters)
        hist, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf = fitted_dist.pdf(bin_centers)
        a = self.alpha
        eps = 1e-10
        p = hist + eps
        q = pdf + eps
        divergence = (1.0 / (a * (a - 1))) * (np.sum(p**a * q**(1 - a)) - 1.0)
        return -divergence


class WassersteinDistance(UnivariateDistributionMetric):
    def evaluate(
        self,
        distribution: UnivariateDistribution,
        parameters,
        data: np.ndarray
    ) -> float:
        """
        1D Earth Mover's Distance between samples and fitted distribution.
        Returns negative distance (higher is better).
        """
        fitted_dist = distribution(*parameters)
        # Generate samples from the theoretical distribution
        num = len(data)
        sampled = fitted_dist.rvs(size=num)
        wd = wasserstein_distance(data, sampled)
        return -wd


class MockUnivariateDistributionMetric(UnivariateDistributionMetric):
    def evaluate(
        self, distribution: UnivariateDistribution, parameters, data: np.ndarray
    ) -> float:
        return 1.0

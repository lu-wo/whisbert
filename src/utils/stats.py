import numpy as np
from scipy import stats


def f_oneway(res_model1, res_model2):
    """Perform one-way ANOVA.

    Args:
        res_model1 (list): List of residuals from model 1.
        res_model2 (list): List of residuals from model 2.

    Returns:
        float: F-statistic.
        float: p-value.
    """
    F, p = stats.f_oneway(res_model1, res_model2)
    return F, p


def shannon_entropy(p, base=2, tol=1e-4):
    """
    Compute the Shannon entropy of each row in the input probability matrix.

    Args:
    p (numpy.ndarray): A 2D array where each row represents a probability distribution.
                       Shape is (N, voc_size).
    tol (float): Tolerance for checking that the probability distributions are normalized.

    Returns:
    numpy.ndarray: A 1D array of Shannon entropy values. Shape is (N,).
    """
    return stats.entropy(p.T, base=base)

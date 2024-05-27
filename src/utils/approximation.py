from scipy.optimize import curve_fit
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from scipy.stats import entropy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma


def knn_differential_entropy(data, k=5):
    """
    Estimate the differential entropy of continuous data using k-nearest neighbors.

    Parameters:
        - data: list or array of continuous values
        - k: the number of neighbors for k-nearest neighbors algorithm

    Returns:
        - entropy: estimated differential entropy of the data
    """
    # Reshape data for compatibility with NearestNeighbors
    data = np.array(data).reshape(-1, 1)

    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(data)

    # Find distances to the k-th nearest neighbor
    distances, _ = nn.kneighbors(data, n_neighbors=k)
    kth_nearest_distances = distances[:, -1]

    # Estimate differential entropy
    n = len(data)
    entropy = (
        digamma(n) - digamma(k) + np.log(np.pi) + np.mean(np.log(kth_nearest_distances))
    )

    return entropy


def continuous_to_shannon(data):
    """
    Estimate the Shannon entropy of continuous data by considering
    the data as discrete samples.

    Parameters:
        - data: list or array of continuous values

    Returns:
        - shannon_entropy: estimated Shannon entropy of the data
    """
    # Calculate the probability mass function
    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)

    # Calculate the Shannon entropy
    shannon_entropy = entropy(probabilities, base=2)

    return shannon_entropy


def cross_validate_gkde_bandwidth(
    train_data,
    test_data,
    parameters=["scott", "silverman", 0.01, 0.1, 0.3],
):
    """Cross validate bandwidth for KDE

    Parameters
    ----------
    func : function
        KDE function to use
    samples : array-like
        Samples to cross validate
    parameters : list
        List of parameters to try

    Returns
    -------
    best_params : float
        Best parameter
    """
    best_params = None
    best_score = float("-inf")
    for param in parameters:
        try:
            kde = gaussian_kde(train_data, bw_method=param)
            score = kde.logpdf(test_data).mean()
            print(f"param {param}, score {score}")
            if score > best_score:
                print(f"new best param {param}, score {score}")
                best_score = score
                best_params = param
        except Exception as e:
            print(e)
            continue
    return best_params


def monte_carlo_diff_entropy(density_func, samples, num_samples=10000):
    """
    Estimate the differential entropy of a density function using Monte Carlo method.

    Parameters:
    - density_func: A function that calculates the density (probability) for given samples.
    - samples: A list or numpy array of samples from the density function.
    - num_samples: The number of samples to draw for the estimation (default 10000).

    Returns:
    - The estimated differential entropy.

    Steps:
    1. Draw a large number of samples from the distribution (density function).
    2. Compute the negative log of the density function evaluated at these points.
    3. The average of these quantities will be our estimate for the differential entropy.
    """

    # Draw a large number of samples from the density function
    if num_samples < len(samples):
        samples = np.random.choice(samples, num_samples, replace=True)

    # Compute the log-density (negative) for each sample
    log_densities = -np.log(density_func(samples))

    # Compute the empirical average of these quantities
    entropy_estimate = np.mean(log_densities)

    return entropy_estimate


def sine_function(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset


def fit_sine_to_signal(signal, maxfev=10000):
    x_data = np.arange(len(signal))

    # Estimate initial parameters for the sine function
    amplitude_est = (np.max(signal) - np.min(signal)) / 2
    frequency_est = 1 / len(signal)
    phase_est = 0
    offset_est = np.mean(signal)

    initial_params = [amplitude_est, frequency_est, phase_est, offset_est]

    try:
        # Fit the sine function to the signal
        params, _ = curve_fit(
            sine_function,
            x_data,
            signal,
            p0=initial_params,
            maxfev=maxfev,
            full_output=True,
        )[:2]
    except RuntimeError as e:
        print("Warning:", e)
        print("Returning the best estimates found so far")
        return None, None

    # Compute the approximated signal using the fitted parameters
    approximated_signal = sine_function(x_data, *params)

    return approximated_signal, params

import numpy as np


def create_some_normal_samples(
    size: int, count: int, a: float, sigma: float
) -> np.array:
    return np.random.normal(a, sigma, (count, size))


def create_sample_norm_vars(
    size: int, length: int, count: int, a: float, sigma: float
) -> np.array:
    mean_variances = list()
    for i in range(size):
        samples = np.random.normal(a, sigma, (count, length))
        sample_vars = np.var(samples, axis=1)
        mean_variances.append(np.mean(sample_vars))
    return np.array(mean_variances)

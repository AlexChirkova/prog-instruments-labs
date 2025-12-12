import numpy as np

from graphs import *
from normal import *
from statistic import *


def main(size_of_samples, count_of_samples, a, sigma, repetitions):

    samples = create_some_normal_samples(size_of_sample, count_of_samples, a, sigma)
    sample_means = np.mean(samples, axis=1)

    print_statistic(sample_means)
    print(f"Теоретическая дисперсия: {(sigma ** 2) / size_of_sample:.4f}")

    show_emp_and_theor_norm_distrib_func(sample_means, a, sigma)
    show_emp_and_theor_norm_distrib_density(sample_means, a, sigma)

    count_above_a = np.sum(sample_means > a)
    proportion_above_a = count_above_a / count_of_samples

    print(
        f"\nКоличество случаев, когда среднее Х > a: {count_above_a} из {count_of_samples}"
    )
    print(f"Доля случаев: {proportion_above_a:.4f}, должно быть 0.5")

    sample_vars = np.var(samples, axis=1)

    print_statistic(sample_vars)
    print(
        f"Теоретическкая дисперсия: {2 * sigma ** 4 * (size_of_sample - 1) / size_of_sample ** 2}"
    )

    show_emp_distrib_func(sample_vars)
    show_emp_distrib_density(sample_vars)

    count_above_sigma = np.sum(sample_vars < sigma**2)
    proportion_above_sigma = count_above_sigma / count_of_samples

    print(
        f"\nКоличество случаев, когда D(x) < sigma^2: {count_above_sigma} из {count_of_samples}"
    )
    print(f"Доля случаев: {proportion_above_sigma:.4f}")

    sample_y = size_of_sample * sample_vars / sigma**2

    print(f"\nХарактеристики случайной величины Y:")
    print_statistic(sample_y)

    print_theor_stat_chi2(sample_vars)

    show_emp_and_theor_chi2_distrib_func(sample_y)
    show_emp_and_theor_norm_distrib_density(sample_y, a, sigma)

    sample_mean_vars = create_sample_norm_vars(
        repetitions, size_of_sample, count_of_samples, a, sigma
    )

    stat_characteristics_mean_vars(sample_mean_vars, sigma)

    boxplot_mean_var(sample_mean_vars)
    hist_mean_var(sample_mean_vars, sigma)


if __name__ == "__main__":
    size_of_sample = 7
    count_of_samples = 120
    a = -2
    sigma = 3
    repetitions = 550
    main(size_of_sample, count_of_samples, a, sigma, repetitions)

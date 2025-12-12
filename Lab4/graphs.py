import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def boxplot_mean_var(sample: np.array) -> None:
    plt.figure(figsize=(10, 6))

    plt.boxplot(
        sample,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="red"),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue"),
        flierprops=dict(marker="o", color="red", alpha=0.5),
    )

    plt.ylabel("Средние значения выборочных дисперсий")
    plt.title(f"Ящичковая диаграмма\n(M = {len(sample)} повторений)")

    plt.grid(True, alpha=0.3)
    plt.show()


def hist_mean_var(sample: np.array, sigma: float) -> None:
    plt.hist(
        sample,
        bins="scott",
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    mean_of_mean_vars = np.mean(sample)
    theoretical_var_of_means = sigma**2 * (len(sample) - 1) / len(sample)

    plt.axvline(
        mean_of_mean_vars,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Среднее = {mean_of_mean_vars:.3f}",
    )
    plt.axvline(
        theoretical_var_of_means,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Теоретическое = {theoretical_var_of_means:.3f}",
    )

    plt.xlabel("Средние значения выборочных дисперсий")
    plt.ylabel("Плотность вероятности")
    plt.title("Гистограмма средних дисперсий")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def show_emp_and_theor_norm_distrib_func(
    sample: np.array, a: float, sigma: float
) -> None:
    sorted_sample = np.sort(sample)

    theoretical_x = np.linspace(sorted_sample[0], sorted_sample[-1], 1000)
    theoretical_cdf = stats.norm.cdf(theoretical_x, a, sigma / np.sqrt(len(sample)))

    plt.figure(figsize=(10, 6))

    plt.hist(sorted_sample, bins="scott", density=True, cumulative=True, label="F^(x)")
    plt.plot(theoretical_x, theoretical_cdf, "r-", linewidth=2, label="F(x)")

    plt.xlabel("X")
    plt.ylabel("F(x)")
    plt.title("F(x) и F^(x)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def show_emp_and_theor_norm_distrib_density(
    sample: np.array, a: float, sigma: float
) -> None:
    plt.figure(figsize=(10, 6))

    _, bins, _ = plt.hist(
        sample, bins="scott", density=True, alpha=0.7, color="lightblue", label="f^(x)"
    )

    theoretical_x = np.linspace(bins[0], bins[-1], 1000)
    theoretical_pdf = stats.norm.pdf(theoretical_x, a, sigma / np.sqrt(len(sample)))

    plt.plot(theoretical_x, theoretical_pdf, "r-", linewidth=2, label="f(x)")

    plt.xlabel("X")
    plt.ylabel("f(x)")
    plt.title("f(x) и f^(x)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def show_emp_distrib_func(sample: np.array) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins="scott", density=True, cumulative=True, label="F^(x)")

    plt.xlabel("X")
    plt.ylabel("F(x)")
    plt.title("F(x)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def show_emp_distrib_density(sample) -> None:
    plt.figure(figsize=(10, 6))
    _, bins, _ = plt.hist(
        sample, bins="scott", density=True, alpha=0.7, color="lightblue", label="f^(x)"
    )

    plt.xlabel("X")
    plt.ylabel("f(x)")
    plt.title("f(x)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def show_emp_and_theor_chi2_distrib_density(sample: np.array, df: int) -> None:
    plt.hist(
        sample,
        bins="scott",
        density=True,
        alpha=0.7,
        color="lightgreen",
        label="Гистограмма Y",
    )

    theoretical_y = np.linspace(0, np.max(sample), 1000)
    theoretical_pdf_y = stats.chi2.pdf(theoretical_y, df)

    plt.plot(
        theoretical_y,
        theoretical_pdf_y,
        "r-",
        linewidth=2,
        label=f"Теоретическая χ²({df})",
    )

    plt.xlabel("Значения Y")
    plt.ylabel("Плотность вероятности")
    plt.title("Гистограмма и теоретическая плотность Y")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def show_emp_and_theor_chi2_distrib_func(sample: np.array, df: int) -> None:
    plt.hist(
        sample,
        bins="scott",
        density=True,
        alpha=0.7,
        color="lightgreen",
        label="F(Y)",
        cumulative=True,
    )

    theoretical_y = np.linspace(0, np.max(sample), 1000)
    theoretical_cdf_y = stats.chi2.cdf(theoretical_y, df)

    plt.plot(
        theoretical_y,
        theoretical_cdf_y,
        "r-",
        linewidth=2,
        label=f"Теоретическая χ²({df})",
    )

    plt.xlabel("Значения Y")
    plt.ylabel("F(y)")
    plt.title("Эмпирическая и теоретическая функции Y")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

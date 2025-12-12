import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def statistic(sample: np.array) -> tuple:
    stat = list()
    stat.append(np.mean(sample))
    stat.append(np.median(sample))
    stat.append(np.std(sample))
    stat.append(np.var(sample, ddof=1))
    stat.append(stats.skew(sample))
    stat.append(stats.kurtosis(sample))
    return tuple(stat)


def print_statistic(sample: np.array) -> None:
    stat = statistic(sample)
    print("Основные выборочные статистические характеристики:")
    print(f"  Среднее: {stat[0]:.4f}")
    print(f"  Медиана: {stat[1]:.4f}")
    print(f"  Стандартное отклонение: {stat[2]:.4f}")
    print(f"  Исправленная дисперсия: {stat[3]:.4f}")
    print(f"  Коэффициент асимметрии: {stat[4]:.4f}")
    print(f"  Коэффициент эксцесса: {stat[5]:.4f}")


def create_some_normal_samples(
    size: int, count: int, a: float, sigma: float
) -> np.array:
    return np.random.normal(a, sigma, (count, size))


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


def theor_statistic_chi2(sample: np.array, df: int = 0) -> tuple:
    if df == 0:
        df = len(sample) - 1

    theor_stat = list()
    theor_stat.append(df)
    theor_stat.append(2 * df)
    theor_stat.append(df * (1 - 2 / (9 * df)) ** 3)
    theor_stat.append(np.sqrt(8 / df))
    theor_stat.append(12 / df)
    return tuple(theor_stat)


def print_theor_stat_chi2(sample: np.array, df: int = 0) -> None:
    theor_stat = theor_statistic_chi2(sample, df)

    print(f"Теоретические характеристики Y ~ χ²({theor_stat[0]}):")
    print(f"   Математическое ожидание: {theor_stat[0]:.4f}")
    print(f"   Дисперсия: {theor_stat[1]:.4f}")
    print(f"   Медиана (приближенно): {theor_stat[2]:.4f}")
    print(f"   Коэффициент асимметрии: {theor_stat[3]:.4f}")
    print(f"   Коэффициент эксцесса: {theor_stat[4]:.4f}")


def show_emp_and_theor_chi2_distrib_density(sample: np.array, df: int = 0) -> None:
    if df == 0:
        df = len(sample) - 1

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


def show_emp_and_theor_chi2_distrib_func(sample: np.array, df: int = 0) -> None:
    if df == 0:
        df = len(sample) - 1

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


def create_sample_norm_vars(
    size: int, length: int, count: int, a: float, sigma: float
) -> np.array:
    mean_variances = list()
    for i in range(size):
        samples = np.random.normal(a, sigma, (count, length))
        sample_vars = np.var(samples, axis=1)
        mean_variances.append(np.mean(sample_vars))
    return np.array(mean_variances)


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


def stat_characteristics_mean_vars(sample: np.array, sigma: float) -> None:
    mean_of_mean_vars = np.mean(sample)
    median_of_mean_vars = np.median(sample)
    theoretical_var_of_means = sigma**2 * (len(sample) - 1) / len(sample)

    print("\nРезультаты:")
    print(f"Среднее средних дисперсий: {mean_of_mean_vars:.4f}")
    print(f"Медиана средних дисперсий: {median_of_mean_vars:.4f}")
    print(f"Теоретическая дисперсия: {theoretical_var_of_means:.4f}")
    print(f"Отклонение от теоретической: {abs(mean_of_mean_vars - sigma**2):.4f}")
    print(
        f"Относительная ошибка: {abs(mean_of_mean_vars - theoretical_var_of_means)/theoretical_var_of_means*100:.2f}%"
    )


def main():
    size_of_sample = 7
    count_of_samples = 120
    a = -2
    sigma = 3

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

    repetitions = 550
    sample_mean_vars = create_sample_norm_vars(
        repetitions, size_of_sample, count_of_samples, a, sigma
    )

    stat_characteristics_mean_vars(sample_mean_vars, sigma)

    boxplot_mean_var(sample_mean_vars)
    hist_mean_var(sample_mean_vars, sigma)

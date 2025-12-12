import numpy as np
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


def theor_statistic_chi2(sample: np.array, df: int) -> tuple:
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

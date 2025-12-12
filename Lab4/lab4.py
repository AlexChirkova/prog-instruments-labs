import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 5й вариант лабы по статистике


# создание масива случайных значений
size_of_simple = 7
count_of_samples = 120
a = -2
sigma = 3
repetitions = 550
samples = np.random.normal(a, sigma, (count_of_samples, size_of_simple))

sample_means = np.mean(samples, axis=1)
print(sample_means)

sorted_means = np.sort(sample_means)

# Эмпирическая и теоретиеская функции распределения
theoretical_x = np.linspace(sorted_means[0], sorted_means[-1], 1000)
theoretical_cdf = stats.norm.cdf(theoretical_x, a, sigma / np.sqrt(size_of_simple))

plt.figure(figsize=(10, 6))
plt.hist(sorted_means, bins='scott', density=True, cumulative=True, label='F^(x)')
plt.plot(theoretical_x, theoretical_cdf, 'r-', linewidth=2, label='F(x)')
plt.xlabel('X')
plt.ylabel('F(x)')
plt.title('F(x) и F^(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Гистограмма
plt.figure(figsize=(10, 6))
_, bins, _ = plt.hist(sample_means, bins='scott', density=True,
                      alpha=0.7, color='lightblue',
                      label='f^(x)')

# Теоретическая плотность распределения
theoretical_x = np.linspace(bins[0], bins[-1], 1000)
theoretical_pdf = stats.norm.pdf(theoretical_x, a, sigma / np.sqrt(size_of_simple))
plt.plot(theoretical_x, theoretical_pdf, 'r-', linewidth=2, label='f(x)')

plt.xlabel('X')
plt.ylabel('f(x)')
plt.title('f(x) и f^(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

mean_of_means = np.mean(sample_means)
median_of_means = np.median(sample_means)
variance_of_means = np.var(sample_means, ddof=1)  # S^2

print(f"\nВыборочное среднее: {mean_of_means:.4f}")
print(f"Выборочная медиана: {median_of_means:.4f}")
print(f"Выборочная дисперсия: {variance_of_means:.4f}")
print(f"Теоретическая дисперсия Х-: {(sigma ** 2) / size_of_simple:.4f}")

skewness = stats.skew(sample_means)
kurtosis = stats.kurtosis(sample_means)

print(f"\nКоэффициент асимметрии: {skewness:.4f}, должно быть 0")
print(f"Коэффициент эксцесса: {kurtosis:.4f}, должно быть 0")

# Подсчет случаев, когда Х- > a
count_above_a = np.sum(sample_means > a)
proportion_above_a = count_above_a / count_of_samples

print(f"\nКоличество случаев, когда Х- > a: {count_above_a} из {count_of_samples}")
print(f"Доля случаев: {proportion_above_a:.4f}, должно быть 0.5")

# ЧАСТЬ 1, про дисперсию
sample_vars = np.var(samples, axis=1)
print(samples[1])
print()
print(sample_vars)

mean_of_vars = np.mean(sample_vars)
median_of_vars = np.median(sample_vars)
variance_of_vars = np.var(sample_vars, ddof=1)  # S^2

print(f"\nВыборочное среднее: {mean_of_vars:.4f}")
print(f"Выборочная медиана: {median_of_vars:.4f}")
print(f"Выборочная дисперсия: {variance_of_vars:.4f}")
print(f"Теоретическкая дисперсия: {2 * sigma ** 4 * (size_of_simple - 1) / size_of_simple ** 2}")

# Эмпирическая функция распределения

plt.figure(figsize=(10, 6))
plt.hist(sample_vars, bins='scott', density=True, cumulative=True, label='F^(x)')
plt.xlabel('X')
plt.ylabel('F(x)')
plt.title('F(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Гистограмма
plt.figure(figsize=(10, 6))
_, bins, _ = plt.hist(sample_vars, bins='scott', density=True,
                      alpha=0.7, color='lightblue',
                      label='f^(x)')
plt.xlabel('X')
plt.ylabel('f(x)')
plt.title('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

count_above_sigma = np.sum(sample_vars < sigma ** 2)
proportion_above_sigma = count_above_sigma / count_of_samples
print(f"\nКоличество случаев, когда D(x) < sigma^2: {count_above_sigma} из {count_of_samples}")
print(f"Доля случаев: {proportion_above_sigma:.4f}")

sample_y = size_of_simple * sample_vars / sigma ** 2

mean_Y = np.mean(sample_y)
median_Y = np.median(sample_y)
variance_Y = np.var(sample_y, ddof=1)
skewness_Y = stats.skew(sample_y)
kurtosis_Y = stats.kurtosis(sample_y)

print(f"\nХарактеристики случайной величины Y:")
print(f"   - Выборочное среднее: {mean_Y:.4f}")
print(f"   - Выборочная медиана: {median_Y:.4f}")
print(f"   - Выборочная дисперсия: {variance_Y:.4f}")
print(f"   - Коэффициент асимметрии: {skewness_Y:.4f}")
print(f"   - Коэффициент эксцесса: {kurtosis_Y:.4f}")

df = size_of_simple - 1  # degrees of freedom

theoretical_mean_Y = df
theoretical_variance_Y = 2 * df
theoretical_median_Y = df * (1 - 2 / (9 * df)) ** 3
theoretical_skewness_Y = np.sqrt(8 / df)
theoretical_kurtosis_Y = 12 / df

print(f"\nТеоретические характеристики Y ~ χ²({df}):")
print(f"   - Математическое ожидание: {theoretical_mean_Y:.4f}")
print(f"   - Дисперсия: {theoretical_variance_Y:.4f}")
print(f"   - Медиана (приближенно): {theoretical_median_Y:.4f}")
print(f"   - Коэффициент асимметрии: {theoretical_skewness_Y:.4f}")
print(f"   - Коэффициент эксцесса: {theoretical_kurtosis_Y:.4f}")

plt.hist(sample_y, bins='scott', density=True,
         alpha=0.7, color='lightgreen',
         label='Гистограмма Y')

theoretical_y = np.linspace(0, np.max(sample_y), 1000)
theoretical_pdf_Y = stats.chi2.pdf(theoretical_y, df)
plt.plot(theoretical_y, theoretical_pdf_Y, 'r-', linewidth=2,
         label=f'Теоретическая χ²({df})')

plt.xlabel('Значения Y')
plt.ylabel('Плотность вероятности')
plt.title('Гистограмма и теоретическая плотность Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.hist(sample_y, bins='scott', density=True,
         alpha=0.7, color='lightgreen',
         label='F(Y)', cumulative=True)

theoretical_y = np.linspace(0, np.max(sample_y), 1000)
theoretical_pdf_Y = stats.chi2.cdf(theoretical_y, df)
plt.plot(theoretical_y, theoretical_pdf_Y, 'r-', linewidth=2,
         label=f'Теоретическая χ²({df})')


plt.xlabel('Значения Y')
plt.ylabel('F(y)')
plt.title('Империческая и теоретическая функция Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#CHAST 2
mean_variances_part2 = []
for i in range(repetitions):
    samples = np.random.normal(a, sigma, (count_of_samples, size_of_simple))
    sample_vars = np.var(samples, axis=1)
    mean_variances_part2.append(np.mean(sample_vars))

mean_variances_part2 = np.array(mean_variances_part2)
print(mean_variances_part2)

plt.figure(figsize=(10, 6))

plt.boxplot(mean_variances_part2, vert=True, patch_artist=True,
           boxprops=dict(facecolor='lightblue', color='blue'),
           medianprops=dict(color='red'),
           whiskerprops=dict(color='blue'),
           capprops=dict(color='blue'),
           flierprops=dict(marker='o', color='red', alpha=0.5))
plt.ylabel('Средние значения выборочных дисперсий')
plt.title(f'Ящичковая диаграмма\n(M = {repetitions} повторений)')
plt.grid(True, alpha=0.3)
plt.show()

mean_of_mean_vars = np.mean(mean_variances_part2)
median_of_mean_vars = np.median(mean_variances_part2)
theoretical_var_of_means = sigma ** 2 * (size_of_simple - 1) / size_of_simple

print("\nРезультаты:")
print(f"Среднее средних дисперсий: {mean_of_mean_vars:.4f}")
print(f"Медиана средних дисперсий: {median_of_mean_vars:.4f}")
print(f"Теоретическая дисперсия: {theoretical_var_of_means:.4f}")
print(f"Отклонение от теоретической: {abs(mean_of_mean_vars - sigma**2):.4f}")
print(f"Относительная ошибка: {abs(mean_of_mean_vars - theoretical_var_of_means)/theoretical_var_of_means*100:.2f}%")

plt.hist(mean_variances_part2, bins='scott', density=True,
                                alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(mean_of_mean_vars, color='red', linestyle='--', linewidth=2,
           label=f'Среднее = {mean_of_mean_vars:.3f}')
plt.axvline(theoretical_var_of_means, color='green', linestyle='--', linewidth=2,
           label=f'Теоретическое = {theoretical_var_of_means:.3f}')
plt.xlabel('Средние значения выборочных дисперсий')
plt.ylabel('Плотность вероятности')
plt.title('Гистограмма средних дисперсий')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

sorted_mean_vars = np.sort(mean_variances_part2)
plt.hist(mean_variances_part2, bins='scott', density=True, cumulative=True, label='F^(x)')
plt.show()
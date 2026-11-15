"""
День 3. Компромисс смещения и дисперсии (Bias-Variance Tradeoff).

Демонстрируем, как сложность модели влияет на смещение и дисперсию.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  КОМПРОМИСС СМЕЩЕНИЯ И ДИСПЕРСИИ")
print("=" * 60)

# ============================================================
# ЧАСТЬ 1: ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================
np.random.seed(42)

# Истинная функция
def true_function(x):
    return np.sin(x) + 0.5 * x

# Генерируем много наборов данных из одного распределения
n_datasets = 20
n_samples = 30
datasets = []

for i in range(n_datasets):
    X = 3 * np.random.rand(n_samples, 1)
    y = true_function(X) + np.random.randn(n_samples, 1) * 0.3
    datasets.append((X, y))

print(f"Сгенерировано {n_datasets} наборов данных по {n_samples} точек")

# Тестовая точка для оценки
X_test = np.linspace(0, 3, 100).reshape(-1, 1)
y_test_true = true_function(X_test)

# ============================================================
# ЧАСТЬ 2: ОБУЧАЕМ МОДЕЛИ НА КАЖДОМ НАБОРЕ ДАННЫХ
# ============================================================
degrees = [1, 3, 9]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, degree in enumerate(degrees):
    predictions = []
    
    for X, y in datasets:
        # Полиномиальные признаки
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_test_poly = poly.transform(X_test)
        
        # Обучение
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Предсказание на тестовых точках
        y_pred = model.predict(X_test_poly)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)  # (n_datasets, n_test_points)
    
    # Среднее предсказание по всем наборам данных
    mean_prediction = predictions.mean(axis=0)
    
    # Дисперсия предсказаний (variability модели)
    variance = predictions.var(axis=0).mean()
    
    # Смещение (насколько среднее предсказание отличается от истины)
    bias = ((mean_prediction - y_test_true.flatten()) ** 2).mean()
    
    # Общая ошибка (примерно)
    total_error = bias + variance
    
    print(f"\nСтепень {degree}:")
    print(f"  Смещение (Bias²):  {bias:.4f}")
    print(f"  Дисперсия (Var):   {variance:.4f}")
    print(f"  Общая ошибка:      {total_error:.4f}")
    
    # Визуализация
    axes[i].plot(X_test, y_test_true, 'k-', linewidth=2, label='Истинная функция')
    axes[i].plot(X_test, mean_prediction, 'r-', linewidth=2, label='Среднее предсказание')
    
    # Показываем несколько предсказаний из разных наборов данных
    for j in range(min(5, n_datasets)):
        axes[i].plot(X_test, predictions[j], 'b-', alpha=0.1, linewidth=0.5)
    
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].set_title(f'Степень {degree}\nBias²={bias:.3f}, Var={variance:.3f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day03_bias_variance.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day03_bias_variance.png")

# ============================================================
# ЧАСТЬ 3: ГРАФИК КОМПРОМИССА
# ============================================================
print("\nСтроим график компромисса...")

all_degrees = list(range(1, 15))
biases = []
variances = []
total_errors = []

for degree in all_degrees:
    predictions = []
    
    for X, y in datasets:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_test_poly = poly.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        y_pred = model.predict(X_test_poly)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    mean_prediction = predictions.mean(axis=0)
    
    bias = ((mean_prediction - y_test_true.flatten()) ** 2).mean()
    variance = predictions.var(axis=0).mean()
    
    biases.append(bias)
    variances.append(variance)
    total_errors.append(bias + variance)

# Визуализация компромисса
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(all_degrees, biases, 'o-', color='blue', label='Смещение² (Bias²)')
ax.plot(all_degrees, variances, 'o-', color='red', label='Дисперсия (Variance)')
ax.plot(all_degrees, total_errors, 'o-', color='green', linewidth=3, label='Общая ошибка')

best_degree = all_degrees[np.argmin(total_errors)]
ax.axvline(x=best_degree, color='gray', linestyle='--', label=f'Оптимум (степень {best_degree})')

ax.set_xlabel('Степень полинома (сложность модели)')
ax.set_ylabel('Ошибка')
ax.set_title('Компромисс смещения и дисперсии')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day03_bias_variance_tradeoff.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day03_bias_variance_tradeoff.png")

# ============================================================
# ЧАСТЬ 4: ВЫВОДЫ
# ============================================================
print("\n" + "=" * 60)
print("  ВЫВОДЫ")
print("=" * 60)

print(f"""
С УВЕЛИЧЕНИЕМ СЛОЖНОСТИ МОДЕЛИ:
  • Смещение (Bias) ↓ — модель лучше улавливает закономерности
  • Дисперсия (Variance) ↑ — модель становится чувствительнее к шуму
  
ОПТИМАЛЬНАЯ СЛОЖНОСТЬ: {best_degree}
  Это точка минимальной суммарной ошибки.

ИНТЕРПРЕТАЦИЯ ГРАФИКА:
  • Слева от оптимума: недообучение (высокое смещение)
  • Справа от оптимума: переобучение (высокая дисперсия)
  • В оптимуме: баланс между смещением и дисперсией

ПРАКТИЧЕСКОЕ ПРАВИЛО:
  Начинай с простой модели. Увеличивай сложность,
  пока ошибка на валидации падает. Останавливайся,
  когда ошибка на валидации начинает расти.
""")
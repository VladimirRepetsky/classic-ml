"""
День 3. Полиномиальная регрессия на синтетических данных.

Генерируем нелинейные данные и пробуем описать их
полиномиальными моделями разной степени.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ")
print("=" * 60)

# ============================================================
# ЧАСТЬ 1: ГЕНЕРАЦИЯ НЕЛИНЕЙНЫХ ДАННЫХ
# ============================================================
print("\nГенерация нелинейных данных...")

np.random.seed(42)

# Истинная зависимость: синусоида + шум
X = 3 * np.random.rand(100, 1)

# ВАЖНО: .ravel() превращает (100, 1) → (100,), то есть в одномерный вектор
y = (np.sin(X) + 0.5 * X + np.random.randn(100, 1) * 0.2).ravel()

print(f"Данных: {X.shape[0]} точек")
print(f"Истинная зависимость: y = sin(x) + 0.5·x + шум")
print(f"Форма X: {X.shape}, форма y: {y.shape}")

# ============================================================
# ЧАСТЬ 2: ЛИНЕЙНАЯ РЕГРЕССИЯ (степень 1)
# ============================================================
print("\n--- Линейная регрессия (степень 1) ---")

model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

mse_linear = mean_squared_error(y, y_pred_linear)
print(f"  MSE: {mse_linear:.4f}")
w = model_linear.coef_[0]
b = model_linear.intercept_
print(f"  Модель: y = {w:.3f}·x + {b:.3f}")

# ============================================================
# ЧАСТЬ 3: ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ (степени 2, 3, 5, 10)
# ============================================================
degrees = [2, 3, 5, 10]
models = {}
predictions = {}
mse_values = {}

for degree in degrees:
    print(f"\n--- Полиномиальная регрессия (степень {degree}) ---")
    
    # Создаём полиномиальные признаки
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    print(f"  Признаков после полиномиализации: {X_poly.shape[1]}")
    
    # Обучаем линейную регрессию на полиномиальных признаках
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Предсказание
    y_pred = model.predict(X_poly)
    
    mse = mean_squared_error(y, y_pred)
    
    models[degree] = model
    predictions[degree] = y_pred
    mse_values[degree] = mse
    
    print(f"  MSE: {mse:.4f}")

# ============================================================
# ЧАСТЬ 4: ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\nВизуализация...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Данные
axes[0].scatter(X, y, alpha=0.5, s=20, color='steelblue', label='Данные')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Исходные данные (нелинейная зависимость)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Линейная модель
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
axes[1].scatter(X, y, alpha=0.3, s=10, color='steelblue', label='Данные')
axes[1].plot(X_plot, model_linear.predict(X_plot), 'r-', linewidth=2, label='Линейная модель')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(f'Степень 1 (линейная) | MSE = {mse_linear:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Полиномиальные модели
for i, degree in enumerate(degrees):
    ax = axes[i + 2]
    
    # Данные
    ax.scatter(X, y, alpha=0.3, s=10, color='steelblue', label='Данные')
    
    # Предсказание на сетке
    X_plot_poly = PolynomialFeatures(degree=degree).fit_transform(X_plot)
    y_plot = models[degree].predict(X_plot_poly)
    
    ax.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Степень {degree}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Степень {degree} | MSE = {mse_values[degree]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day03_poly_degrees.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day03_poly_degrees.png")

# ============================================================
# ЧАСТЬ 5: СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================
print("\n" + "=" * 60)
print("  СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

print(f"\n{'Степень':<10} {'MSE':>10} {'Количество признаков':>22}")
print("-" * 45)
print(f"{1:<10} {mse_linear:>10.4f} {2:>22}")
for degree in degrees:
    n_features = 1 + degree  # +1 для свободного члена
    print(f"{degree:<10} {mse_values[degree]:>10.4f} {n_features:>22}")

print(f"""
ВЫВОДЫ:
  • Степень 1 (линейная): недообучение — модель слишком простая
  • Степень 2-3: хорошее приближение
  • Степень 5: начинает переобучаться
  • Степень 10: сильное переобучение — модель проходит через каждую точку
  
  Полиномиальная регрессия — это ВСЁ ЕЩЁ линейная модель.
  Мы просто добавили новые признаки (x², x³, ...).
""")
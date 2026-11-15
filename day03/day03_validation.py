"""
День 3. Валидационная выборка для выбора модели.

Учимся правильно выбирать гиперпараметры (степень полинома)
с помощью валидационной выборки.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  ВАЛИДАЦИОННАЯ ВЫБОРКА")
print("=" * 60)

# ============================================================
# ЧАСТЬ 1: ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================
np.random.seed(42)

X = 3 * np.random.rand(200, 1)
y = np.sin(X) + 0.5 * X + np.random.randn(200, 1) * 0.3

print(f"Данных: {X.shape[0]} точек")

# ============================================================
# ЧАСТЬ 2: РАЗДЕЛЕНИЕ НА ТРИ ВЫБОРКИ
# ============================================================
print("\n" + "=" * 60)
print("  РАЗДЕЛЕНИЕ ДАННЫХ")
print("=" * 60)

# Разделяем на обучающую и временную (60/40)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Разделяем временную на валидационную и тестовую (50/50)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Обучающая выборка:   {X_train.shape[0]} точек ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Валидационная выборка: {X_val.shape[0]} точек ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Тестовая выборка:    {X_test.shape[0]} точек ({X_test.shape[0]/len(X)*100:.0f}%)")

print(f"""
ЗАЧЕМ НУЖНЫ ТРИ ВЫБОРКИ?

  ОБУЧАЮЩАЯ (train):
    Модель учится на этих данных.
    
  ВАЛИДАЦИОННАЯ (validation):
    Модель НЕ видит эти данные при обучении.
    Мы используем их для ВЫБОРА модели (степени полинома).
    Это «экзамен» для выбора лучшей модели.
    
  ТЕСТОВАЯ (test):
    Модель НЕ видит эти данные при обучении И при выборе.
    Мы используем их ОДИН РАЗ для финальной оценки.
    Это «выпускной экзамен».
    
  ПРАВИЛО:
    Никогда не подбирай гиперпараметры по тестовой выборке!
    Тестовая выборка — это симуляция «новых данных из реального мира».
""")

# ============================================================
# ЧАСТЬ 3: ПЕРЕБОР СТЕПЕНЕЙ ПОЛИНОМА НА ВАЛИДАЦИИ
# ============================================================
print("\n" + "=" * 60)
print("  ПЕРЕБОР ГИПЕРПАРАМЕТРОВ")
print("=" * 60)

degrees = range(1, 15)
val_errors = []
test_errors = []

for degree in degrees:
    # Полиномиальные признаки
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)
    
    # Обучение
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Ошибки на валидации и тесте
    val_pred = model.predict(X_val_poly)
    test_pred = model.predict(X_test_poly)
    
    val_mse = mean_squared_error(y_val, val_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    val_errors.append(val_mse)
    test_errors.append(test_mse)
    
    print(f"Степень {degree:>2}: Val MSE = {val_mse:.4f}, Test MSE = {test_mse:.4f}")

# ============================================================
# ЧАСТЬ 4: ВЫБОР ЛУЧШЕЙ МОДЕЛИ
# ============================================================
best_degree = degrees[np.argmin(val_errors)]
best_val_mse = min(val_errors)

print(f"\n" + "=" * 60)
print(f"  ЛУЧШАЯ МОДЕЛЬ")
print(f"=" * 60)
print(f"Лучшая степень полинома (по валидации): {best_degree}")
print(f"MSE на валидации: {best_val_mse:.4f}")
print(f"MSE на тесте:     {test_errors[best_degree-1]:.4f}")

# ============================================================
# ЧАСТЬ 5: ВИЗУАЛИЗАЦИЯ ВАЛИДАЦИОННОЙ КРИВОЙ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ошибка от степени полинома
axes[0].plot(degrees, val_errors, 'o-', color='steelblue', label='Ошибка на валидации')
axes[0].plot(degrees, test_errors, 'o-', color='red', label='Ошибка на тесте')
axes[0].axvline(x=best_degree, color='gray', linestyle='--', label=f'Лучшая степень: {best_degree}')
axes[0].set_xlabel('Степень полинома')
axes[0].set_ylabel('MSE')
axes[0].set_title('Валидационная кривая: ошибка от степени полинома')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Лучшая модель
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_poly, y_train)

X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

axes[1].scatter(X_train, y_train, alpha=0.3, s=10, color='steelblue', label='Обучающие данные')
axes[1].scatter(X_val, y_val, alpha=0.5, s=20, color='orange', label='Валидационные данные')
axes[1].scatter(X_test, y_test, alpha=0.5, s=20, color='red', label='Тестовые данные')
axes[1].plot(X_plot, y_plot, 'g-', linewidth=2, label=f'Лучшая модель (степень {best_degree})')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Лучшая модель по валидационной выборке')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day03_validation_curve.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day03_validation_curve.png")

# ============================================================
# ЧАСТЬ 6: АЛЬТЕРНАТИВА — КРОСС-ВАЛИДАЦИЯ
# ============================================================
print("\n" + "=" * 60)
print("  АЛЬТЕРНАТИВА: КРОСС-ВАЛИДАЦИЯ")
print("=" * 60)

from sklearn.model_selection import cross_val_score

cv_errors = []
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_errors.append(-scores.mean())

best_degree_cv = degrees[np.argmin(cv_errors)]

print(f"\nКросс-валидация (5 фолдов):")
print(f"Лучшая степень полинома: {best_degree_cv}")
print(f"Средний MSE: {min(cv_errors):.4f}")

print(f"""
КРОСС-ВАЛИДАЦИЯ vs ВАЛИДАЦИОННАЯ ВЫБОРКА:

  ВАЛИДАЦИОННАЯ ВЫБОРКА:
    Простая и быстрая.
    Но использует только часть данных для обучения.
    Результат может зависеть от конкретного разделения.
    
  КРОСС-ВАЛИДАЦИЯ:
    Более надёжная оценка.
    Каждый образец используется и для обучения, и для валидации.
    Но вычислительно дороже.
    
  ПРАКТИЧЕСКОЕ ПРАВИЛО:
    Для небольших данных → кросс-валидация
    Для больших данных → валидационная выборка
""")
"""
День 2. Метод наименьших квадратов: реализация вручную.

Реализуем нормальное уравнение и сравниваем со встроенным в sklearn.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "california_housing.csv"

print("=" * 60)
print("  МНК ВРУЧНУЮ НА NUMPY")
print("=" * 60)

# Загрузка данных
df = pd.read_csv(CSV_PATH)
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values
feature_names = [c for c in df.columns if c != 'MedHouseVal']

print(f"\nДанные: {X.shape[0]} образцов, {X.shape[1]} признаков")

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Обучающая выборка: {X_train.shape[0]}")
print(f"Тестовая выборка:  {X_test.shape[0]}")

# ============================================================
# ЧАСТЬ 1: НОРМАЛЬНОЕ УРАВНЕНИЕ ВРУЧНУЮ
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 1: НОРМАЛЬНОЕ УРАВНЕНИЕ ВРУЧНУЮ")
print("=" * 60)

# Шаг 1: Добавляем столбец единиц для свободного члена
n_samples = X_train.shape[0]
X_train_b = np.hstack([np.ones((n_samples, 1)), X_train])

print(f"\nИсходная матрица признаков: {X_train.shape}")
print(f"С добавленным столбцом единиц: {X_train_b.shape}")
print(f"Первый столбец (единицы): {X_train_b[:5, 0]}")

# Шаг 2: Вычисляем нормальное уравнение
#   w = (XᵀX)⁻¹ · Xᵀy
#   Используем solve вместо обратной матрицы для устойчивости

print("\nВычисление нормального уравнения:")
print("  1. XᵀX:")
XTX = X_train_b.T @ X_train_b
print(f"     Форма: {XTX.shape}")

print("  2. Xᵀy:")
XTy = X_train_b.T @ y_train
print(f"     Форма: {XTy.shape}")

print("  3. Решение системы (XᵀX)·w = Xᵀy:")
w = np.linalg.solve(XTX, XTy)

print(f"\nВектор весов:")
print(f"  Свободный член (bias): {w[0]:.4f}")
print(f"  Коэффициенты признаков: {w[1:]}")

# Извлекаем коэффициенты и свободный член
coefs_manual = w[1:]
bias_manual = w[0]

# ============================================================
# ЧАСТЬ 2: СРАВНЕНИЕ СО SKLEARN
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 2: СРАВНЕНИЕ СО SKLEARN")
print("=" * 60)

# Обучаем модель из Дня 1
model = LinearRegression()
model.fit(X_train, y_train)

coefs_sklearn = model.coef_
bias_sklearn = model.intercept_

print(f"\n{'Признак':<12} {'МНК вручную':>14} {'sklearn':>12} {'Разница':>12}")
print("-" * 54)
for i, name in enumerate(feature_names):
    diff = abs(coefs_manual[i] - coefs_sklearn[i])
    print(f"{name:<12} {coefs_manual[i]:>14.6f} {coefs_sklearn[i]:>12.6f} {diff:>12.2e}")

print(f"\n{'Bias':<12} {bias_manual:>14.6f} {bias_sklearn:>12.6f} {abs(bias_manual - bias_sklearn):>12.2e}")

# Проверка: предсказания должны совпадать
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
y_pred_manual = X_test_b @ w
y_pred_sklearn = model.predict(X_test)

mse_diff = mean_squared_error(y_pred_manual, y_pred_sklearn)
print(f"\nMSE между предсказаниями двух методов: {mse_diff:.2e}")

if mse_diff < 1e-10:
    print("✅ МНК вручную и sklearn дают одинаковые результаты!")
else:
    print(f"⚠️ Разница {mse_diff:.2e} (нормально, если < 1e-6)")

# ============================================================
# ЧАСТЬ 3: АЛЬТЕРНАТИВА ЧЕРЕЗ np.linalg.lstsq
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 3: РЕШЕНИЕ ЧЕРЕЗ np.linalg.lstsq")
print("=" * 60)
print("""
На практике вместо формулы нормального уравнения используют
функцию наименьших квадратов из NumPy:
""")

w_lstsq, residuals, rank, sv = np.linalg.lstsq(X_train_b, y_train, rcond=None)

print(f"Веса через np.linalg.lstsq: {w_lstsq[:4]}...")
print(f"Свободный член: {w_lstsq[0]:.4f}")

# Сравнение с нормальным уравнением
diff = np.max(np.abs(w - w_lstsq))
print(f"\nМаксимальная разница с нормальным уравнением: {diff:.2e}")

print(f"""
ВЫВОДЫ:
  1. Нормальное уравнение даёт ТОЧНОЕ аналитическое решение
  2. Это то, что делает внутри себя LinearRegression()
  3. На практике используют np.linalg.lstsq (численно устойчивее)
  4. Матрица (XᵀX) может быть вырождена при мультиколлинеарности
     → тогда solve не работает, но работает lstsq
""")
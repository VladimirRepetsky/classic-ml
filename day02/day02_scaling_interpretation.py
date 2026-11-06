"""
День 2. Масштабирование признаков и интерпретация весов.

Почему веса нельзя сравнивать напрямую и как это исправить.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "california_housing.csv"

print("=" * 60)
print("  МАСШТАБИРОВАНИЕ И ИНТЕРПРЕТАЦИЯ ВЕСОВ")
print("=" * 60)

# Загрузка данных
df = pd.read_csv(CSV_PATH)
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values
feature_names = [c for c in df.columns if c != 'MedHouseVal']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# ЧАСТЬ 1: ВЕСА БЕЗ МАСШТАБИРОВАНИЯ (из Дня 1)
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 1: ВЕСА БЕЗ МАСШТАБИРОВАНИЯ")
print("=" * 60)

model_raw = LinearRegression()
model_raw.fit(X_train, y_train)
coefs_raw = model_raw.coef_

print(f"\n{'Признак':<12} {'Вес':>12} {'Масштаб':>12}")
print("-" * 40)
for i, name in enumerate(feature_names):
    # Диапазон значений признака
    scale = X_train[:, i].max() - X_train[:, i].min()
    print(f"{name:<12} {coefs_raw[i]:>12.6f} {scale:>12.2f}")

print(f"""
ПРОБЛЕМА:
  • Веса имеют разный масштаб
  • Признаки имеют разный диапазон значений
  • Нельзя сказать, какой признак важнее, сравнивая веса напрямую
  • Пример: вес для MedInc ≈ 0.44, а для Population ≈ -0.000003
    Это не значит, что доход в 100000 раз важнее!
""")

# ============================================================
# ЧАСТЬ 2: ВЕСА ПОСЛЕ СТАНДАРТИЗАЦИИ
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 2: ВЕСА ПОСЛЕ СТАНДАРТИЗАЦИИ")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
coefs_scaled = model_scaled.coef_

print(f"\n{'Признак':<12} {'Вес (станд.)':>14}")
print("-" * 28)
for i, name in enumerate(feature_names):
    print(f"{name:<12} {coefs_scaled[i]:>14.6f}")

print(f"""
ИНТЕРПРЕТАЦИЯ ПОСЛЕ СТАНДАРТИЗАЦИИ:
  • Все признаки имеют среднее 0 и стд. отклонение 1
  • Веса показывают силу влияния в единицах стандартного отклонения
  • Можно сравнивать веса напрямую!
  • Больший модуль веса → признак важнее
""")

# ============================================================
# ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\nВизуализация...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Веса без стандартизации
colors_raw = ['coral' if c < 0 else 'steelblue' for c in coefs_raw]
axes[0].barh(feature_names, coefs_raw, color=colors_raw)
axes[0].set_xlabel('Вес')
axes[0].set_title('Веса без стандартизации\n(нельзя сравнивать напрямую)')
axes[0].axvline(x=0, color='black', linewidth=0.5)

# Веса после стандартизации
colors_scaled = ['coral' if c < 0 else 'steelblue' for c in coefs_scaled]
axes[1].barh(feature_names, coefs_scaled, color=colors_scaled)
axes[1].set_xlabel('Вес (стандартизованный)')
axes[1].set_title('Веса после стандартизации\n(можно сравнивать!)')
axes[1].axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day02_weights_comparison.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day02_weights_comparison.png")

# ============================================================
# ЧАСТЬ 4: ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ ВЕСОВ
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 4: ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ ВЕСОВ")
print("=" * 60)

print(f"""
Если X_scaled = (X - mean) / std, то модель:

  y = w_scaled · X_scaled + b_scaled

может быть преобразована к исходному масштабу:

  y = w_scaled · (X - mean) / std + b_scaled
    = (w_scaled / std) · X + (b_scaled - w_scaled · mean / std)

То есть:
  w_исходный = w_scaled / std
  b_исходный = b_scaled - Σ(w_scaled · mean / std)
""")

# Обратное преобразование
coefs_original = coefs_scaled / scaler.scale_
bias_original = model_scaled.intercept_ - np.sum(coefs_scaled * scaler.mean_ / scaler.scale_)

print(f"\nПроверка: обратное преобразование должно дать те же веса, что и на сырых данных:")
print(f"\n{'Признак':<12} {'МНК (сырые)':>12} {'Обратное преобр.':>18} {'Разница':>12}")
print("-" * 58)
for i, name in enumerate(feature_names):
    diff = abs(coefs_raw[i] - coefs_original[i])
    print(f"{name:<12} {coefs_raw[i]:>12.6f} {coefs_original[i]:>18.6f} {diff:>12.2e}")

print(f"\n{'Bias':<12} {model_raw.intercept_:>12.6f} {bias_original:>18.6f} {abs(model_raw.intercept_ - bias_original):>12.2e}")

print(f"""
ВЫВОД:
  ✅ Веса после обратного преобразования совпадают с МНК на сырых данных
  ✅ Это доказывает, что стандартизация не меняет модель
  ✅ Она меняет только масштаб, в котором мы измеряем веса
  ✅ Для интерпретации лучше использовать стандартизованные веса
""")
"""
День 3. Итоговая практика: полиномиальная регрессия на California Housing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DAY_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "california_housing.csv"

print("=" * 60)
print("  ПРАКТИКА: POLYNOMIAL REGRESSION НА CALIFORNIA HOUSING")
print("=" * 60)

# Загрузка данных
df = pd.read_csv(CSV_PATH)
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values

print(f"\nДанных: {X.shape[0]} строк, {X.shape[1]} признаков")

# ============================================================
# ЗАДАНИЕ 1: Линейная регрессия (базовая модель)
# ============================================================
print("\n" + "=" * 60)
print("  ЗАДАНИЕ 1: БАЗОВАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_linear = LinearRegression()
model_linear.fit(X_train_scaled, y_train)

y_pred_linear = model_linear.predict(X_test_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)

print(f"\nЛинейная регрессия:")
print(f"  MSE на тесте: {mse_linear:.4f}")
print(f"  R² на тесте:  {model_linear.score(X_test_scaled, y_test):.4f}")

# ============================================================
# ЗАДАНИЕ 2: Полиномиальная регрессия на одном признаке
# ============================================================
print("\n" + "=" * 60)
print("  ЗАДАНИЕ 2: ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ НА ОДНОМ ПРИЗНАКЕ")
print("=" * 60)

# Берём только самый важный признак — MedInc
X_single = df[['MedInc']].values
y = df['MedHouseVal'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

# Перебираем степени полинома
degrees = range(1, 8)
val_errors = []

for degree in degrees:
    # Разделяем обучающую выборку на тренинг и валидацию
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Масштабирование
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    
    # Полиномиальная регрессия
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_tr_scaled, y_tr)
    
    # Ошибка на валидации
    val_pred = model.predict(X_val_scaled)
    val_mse = mean_squared_error(y_val, val_pred)
    val_errors.append(val_mse)
    
    print(f"Степень {degree}: Val MSE = {val_mse:.4f}")

best_degree = degrees[np.argmin(val_errors)]
print(f"\nЛучшая степень полинома: {best_degree}")

# ============================================================
# ЗАДАНИЕ 3: ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\n" + "=" * 60)
print("  ЗАДАНИЕ 3: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Валидационная кривая
axes[0].plot(degrees, val_errors, 'o-', color='steelblue')
axes[0].axvline(x=best_degree, color='gray', linestyle='--', label=f'Лучшая степень: {best_degree}')
axes[0].set_xlabel('Степень полинома')
axes[0].set_ylabel('MSE на валидации')
axes[0].set_title('Валидационная кривая: MedInc → MedHouseVal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Лучшая модель
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
model.fit(X_train_scaled, y_train)

# Сетка для визуализации
X_plot = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_plot = model.predict(X_plot_scaled)

axes[1].scatter(X_train, y_train, alpha=0.1, s=5, color='steelblue', label='Обучающие данные')
axes[1].scatter(X_test, y_test, alpha=0.3, s=10, color='red', label='Тестовые данные')
axes[1].plot(X_plot, y_plot, 'g-', linewidth=2, label=f'Полиномиальная модель (степень {best_degree})')
axes[1].set_xlabel('MedInc (медианный доход)')
axes[1].set_ylabel('MedHouseVal (цена дома)')
axes[1].set_title('Полиномиальная регрессия: доход → цена дома')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day03_housing_poly.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day03_housing_poly.png")

# ============================================================
# ЗАДАНИЕ 4: СРАВНЕНИЕ НА ТЕСТЕ
# ============================================================
print("\n" + "=" * 60)
print("  ЗАДАНИЕ 4: ФИНАЛЬНОЕ СРАВНЕНИЕ НА ТЕСТЕ")
print("=" * 60)

# Линейная модель на одном признаке
model_linear_1 = LinearRegression()
model_linear_1.fit(X_train_scaled, y_train)
y_pred_linear_1 = model_linear_1.predict(X_test_scaled)
mse_linear_1 = mean_squared_error(y_test, y_pred_linear_1)

# Полиномиальная модель
y_pred_poly = model.predict(X_test_scaled)
mse_poly = mean_squared_error(y_test, y_pred_poly)

print(f"\n{'Модель':<30} {'MSE на тесте':>14}")
print("-" * 46)
print(f"{'Линейная (1 признак)':<30} {mse_linear_1:>14.4f}")
print(f"{'Полиномиальная (степень ' + str(best_degree) + ')':<30} {mse_poly:>14.4f}")
print(f"{'Улучшение':<30} {mse_linear_1 - mse_poly:>14.4f}")

print(f"""
ВЫВОДЫ:
  • Полиномиальная регрессия лучше описывает нелинейную зависимость
    между доходом и ценой дома.
  • При низком доходе цена растёт быстро, при высоком — медленнее.
  • Полиномиальная модель улавливает этот эффект.
  
  Но помни:
  • Полиномиальные признаки нужно масштабировать!
  • Степень полинома подбирается через валидацию.
  • Слишком большая степень → переобучение.
""")

# ============================================================
# ЗАДАНИЕ 5: ИТОГИ ДНЯ
# ============================================================
print("\n" + "=" * 60)
print("  ИТОГИ ДНЯ 3")
print("=" * 60)

print(f"""
✅ Понимаю, что полиномиальная регрессия — это линейная модель
   с полиномиальными признаками

✅ Использую PolynomialFeatures для создания признаков

✅ Понимаю переобучение и недообучение

✅ Знаю, как диагностировать переобучение:
   Train MSE << Test MSE → переобучение

✅ Использую кривые обучения для диагностики

✅ Понимаю компромисс смещения и дисперсии

✅ Использую валидационную выборку для выбора гиперпараметров

✅ Знаю, что тестовую выборку используем ОДИН РАЗ

✅ Применил полиномиальную регрессию на реальных данных

ЗАВТРА — День 4: Регуляризация (Ridge, Lasso).
Регуляризация — это способ борьбы с переобучением
без уменьшения сложности модели.
""")
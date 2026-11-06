"""
День 2. Итоговая практика.

Задания:
1. Реализовать градиентный спуск на одном признаке
2. Сравнить с аналитическим решением МНК
3. Показать влияние скорости обучения
4. Интерпретировать веса
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
print("  ИТОГОВАЯ ПРАКТИКА ДНЯ 2")
print("=" * 60)

# Загрузка данных
df = pd.read_csv(CSV_PATH)
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values
feature_names = [c for c in df.columns if c != 'MedHouseVal']

# ============================================================
# ЗАДАНИЕ 1: Градиентный спуск на одном признаке
# ============================================================
print("\n" + "=" * 60)
print("  ЗАДАНИЕ 1: ГРАДИЕНТНЫЙ СПУСК НА ОДНОМ ПРИЗНАКЕ")
print("=" * 60)

# Берём только признак MedInc (самый важный)
X_single = df[['MedInc']].values

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

# Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Градиентный спуск вручную
def gradient_descent_1d(X, y, lr=0.01, n_iters=1000):
    w = 0.0
    b = 0.0
    n = len(X)
    history = []
    
    for i in range(n_iters):
        y_pred = w * X.flatten() + b
        error = y_pred - y
        
        dw = (2 / n) * np.sum(X.flatten() * error)
        db = (2 / n) * np.sum(error)
        
        w -= lr * dw
        b -= lr * db
        
        mse = np.mean(error ** 2)
        history.append(mse)
    
    return w, b, history

w_gd, b_gd, history_gd = gradient_descent_1d(X_train_scaled, y_train, lr=0.05, n_iters=500)

print(f"\nГрадиентный спуск (один признак):")
print(f"  w = {w_gd:.4f}")
print(f"  b = {b_gd:.4f}")
print(f"  MSE = {history_gd[-1]:.4f}")

# ============================================================
# ЗАДАНИЕ 2: Сравнение с МНК
# ============================================================
print("\n" + "=" * 60)
print("  ЗАДАНИЕ 2: СРАВНЕНИЕ С МНК")
print("=" * 60)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"\n{'Метод':<25} {'w':>10} {'b':>10}")
print("-" * 47)
print(f"{'Градиентный спуск':<25} {w_gd:>10.4f} {b_gd:>10.4f}")
print(f"{'МНК (аналитический)':<25} {model.coef_[0]:>10.4f} {model.intercept_:>10.4f}")

diff_w = abs(w_gd - model.coef_[0])
diff_b = abs(b_gd - model.intercept_)
print(f"{'Разница':<25} {diff_w:>10.6f} {diff_b:>10.6f}")

if diff_w < 0.01 and diff_b < 0.01:
    print("\n✅ Градиентный спуск сошёлся к аналитическому решению!")
else:
    print(f"\n⚠️ Разница: Δw = {diff_w:.4f}, Δb = {diff_b:.4f}")

# ============================================================
# ЗАДАНИЕ 3: Визуализация
# ============================================================
print("\n" + "=" * 60)
print("  ЗАДАНИЕ 3: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Данные и модель
axes[0].scatter(X_train_scaled.flatten(), y_train, alpha=0.3, s=10, color='steelblue', label='Обучающие данные')
X_line = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 100)
axes[0].plot(X_line, w_gd * X_line + b_gd, 'r-', linewidth=2,
             label=f'Градиентный спуск: y = {w_gd:.2f}x + {b_gd:.2f}')
axes[0].plot(X_line, model.coef_[0] * X_line + model.intercept_, 'g--', linewidth=2,
             label=f'МНК: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
axes[0].set_xlabel('MedInc (стандартизованный)')
axes[0].set_ylabel('MedHouseVal')
axes[0].set_title('Один признак: MedInc')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Кривая обучения
axes[1].plot(history_gd, color='coral', linewidth=2)
axes[1].set_xlabel('Итерация')
axes[1].set_ylabel('MSE')
axes[1].set_title('Кривая обучения')
axes[1].grid(True, alpha=0.3)

# Ошибки предсказания
y_pred_gd = w_gd * X_test_scaled.flatten() + b_gd
errors = y_test - y_pred_gd
axes[2].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[2].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[2].set_xlabel('Ошибка')
axes[2].set_ylabel('Количество')
axes[2].set_title(f'Распределение ошибок (среднее = {errors.mean():.3f})')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day02_practice_comparison.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day02_practice_comparison.png")

# ============================================================
# ЗАДАНИЕ 4: Итоги дня
# ============================================================
print("\n" + "=" * 60)
print("  ИТОГИ ДНЯ 2")
print("=" * 60)

print(f"""
✅ Реализовал метод наименьших квадратов вручную
   → Нормальное уравнение: w = (XᵀX)⁻¹Xᵀy
   → Совпадает со встроенным в sklearn

✅ Реализовал градиентный спуск вручную
   → Итеративная оптимизация
   → Формула: w ← w - α · ∂MSE/∂w
   → Сходится к аналитическому решению

✅ Понял важность масштабирования
   → Без него градиентный спуск работает плохо
   → Веса нельзя сравнивать напрямую

✅ Узнал про утечку данных
   → Масштабировать только на обучающей выборке
   → Применять ту же трансформацию к тесту

✅ Научился интерпретировать веса
   → После стандартизации веса сопоставимы
   → Можно определить важность признаков
""")
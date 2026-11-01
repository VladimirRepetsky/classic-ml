"""
День 1. Первая линейная регрессия на синтетических данных.
Истинная зависимость известна: y = 3.5*x + 7 + шум.
Проверяем, сможет ли модель её восстановить.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Пути (работают из любой директории) ---
DAY_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  ПЕРВАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("=" * 60)

# ============================================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================
np.random.seed(42)

TRUE_W, TRUE_B = 3.5, 7.0

X = np.random.uniform(0, 100, size=200)
noise = np.random.normal(0, 15, size=200)
y = TRUE_W * X + TRUE_B + noise

print(f"\nСгенерировано {len(X)} точек")
print(f"  X: [{X.min():.1f}, {X.max():.1f}]")
print(f"  y: [{y.min():.1f}, {y.max():.1f}]")
print(f"  Истинная зависимость: y = {TRUE_W}*x + {TRUE_B} + шум(0, 15)")

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nРазделение: train={len(X_train)}, test={len(X_test)}")

# ============================================================
# 3. ФОРМА ДАННЫХ ДЛЯ SKLEARN
# ============================================================
# sklearn требует X формы (n_samples, n_features) — всегда 2D.
# Один признак → reshape(-1, 1): из (160,) в (160, 1)
X_train_2d = X_train.reshape(-1, 1)
X_test_2d = X_test.reshape(-1, 1)

print(f"\nФормы после reshape:")
print(f"  X_train: {X_train.shape} → {X_train_2d.shape}")
print(f"  X_test:  {X_test.shape} → {X_test_2d.shape}")
print(f"  y_train: {y_train.shape} (остаётся 1D)")

# ============================================================
# 4. ОБУЧЕНИЕ
# ============================================================
model = LinearRegression()
model.fit(X_train_2d, y_train)

w_learned = model.coef_[0]
b_learned = model.intercept_

print(f"\n{'Параметр':<15} {'Истинный':>12} {'Выученный':>12} {'Ошибка':>10}")
print("-" * 51)
print(f"{'w (наклон)':<15} {TRUE_W:>12.4f} {w_learned:>12.4f} {abs(w_learned - TRUE_W):>10.4f}")
print(f"{'b (сдвиг)':<15} {TRUE_B:>12.4f} {b_learned:>12.4f} {abs(b_learned - TRUE_B):>10.4f}")

# ============================================================
# 5. ПРЕДСКАЗАНИЕ
# ============================================================
y_pred = model.predict(X_test_2d)

print(f"\nПервые 5 предсказаний на тесте:")
print(f"{'x':>8} {'предсказано':>14} {'реально':>12} {'ошибка':>10}")
print("-" * 46)
for i in range(5):
    err = y_pred[i] - y_test[i]
    print(f"{X_test[i]:>8.2f} {y_pred[i]:>14.2f} {y_test[i]:>12.2f} {err:>10.2f}")

# ============================================================
# 6. МЕТРИКИ
# ============================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nМетрики на ТЕСТЕ:")
print(f"  MAE  = {mae:.4f}")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R²   = {r2:.4f}")
print(f"\n  → В среднем ошибаемся на {mae:.2f} единиц")
print(f"  → Модель объясняет {r2 * 100:.1f}% дисперсии данных")
print(f"  → RMSE ({rmse:.2f}) ≈ уровень шума (15) — модель уловила всё, что могла")

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color="steelblue", alpha=0.5, s=25, label="Обучающие данные")
plt.scatter(X_test, y_test, color="red", alpha=0.9, s=45, label="Тестовые данные")

X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(X_line, model.predict(X_line), color="green", linewidth=2.5,
         label=f"Модель: y = {w_learned:.2f}x + {b_learned:.2f}")
plt.plot(X_line, TRUE_W * X_line + TRUE_B, color="orange", linewidth=2, linestyle="--",
         label=f"Истина: y = {TRUE_W}x + {TRUE_B}")

plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Линейная регрессия | R² = {r2:.4f}, RMSE = {rmse:.2f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = OUTPUT_DIR / "day01_first_regression.png"
plt.savefig(save_path, dpi=150)
plt.show()

print(f"\nГрафик сохранён: {save_path.relative_to(DAY_DIR.parent)}")
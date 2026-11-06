"""
День 2. Влияние скорости обучения на градиентный спуск.

Слишком большая — расходится.
Слишком маленькая — медленно сходится.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
print("  ВЛИЯНИЕ СКОРОСТИ ОБУЧЕНИЯ")
print("=" * 60)

# Загрузка данных
df = pd.read_csv(CSV_PATH)
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values

# Разделение и стандартизация
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Функция градиентного спуска
def gradient_descent(X, y, lr=0.01, n_iters=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    history = []
    
    for i in range(n_iters):
        y_pred = X @ w + b
        error = y_pred - y
        
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)
        
        w -= lr * dw
        b -= lr * db
        
        mse = np.mean(error ** 2)
        history.append(mse)
        
        # Проверка на расходимость
        if np.isnan(mse) or np.isinf(mse) or mse > 1e10:
            print(f"  ⚠️ Расходимость на итерации {i}")
            break
    
    return w, b, history

# ============================================================
# ТЕСТ РАЗНЫХ СКОРОСТЕЙ ОБУЧЕНИЯ
# ============================================================
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
n_iters = 200

print(f"\nТестируем скорости обучения: {learning_rates}")
print(f"Итераций: {n_iters}\n")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

results = []

for idx, lr in enumerate(learning_rates):
    print(f"Скорость обучения: {lr}")
    
    w, b, history = gradient_descent(X_train_scaled, y_train, lr=lr, n_iters=n_iters)
    
    if len(history) > 0 and not np.isnan(history[-1]):
        print(f"  Финальное MSE: {history[-1]:.4f}")
        results.append((lr, history[-1], len(history)))
    else:
        print(f"  ❌ Расходимость!")
        results.append((lr, float('inf'), len(history)))
    
    # Визуализация
    axes[idx].plot(history, linewidth=2, label=f'lr = {lr}')
    axes[idx].set_xlabel('Итерация')
    axes[idx].set_ylabel('MSE')
    axes[idx].set_title(f'Скорость обучения: {lr}')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

# Последняя ось — сводный график
for lr, final_mse, iters in results:
    if final_mse < 1e6:
        axes[-1].scatter(lr, final_mse, s=100, label=f'{lr}')

axes[-1].set_xlabel('Скорость обучения')
axes[-1].set_ylabel('Финальное MSE')
axes[-1].set_title('Итог: скорость обучения vs качество')
axes[-1].set_xscale('log')
axes[-1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day02_learning_rate.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day02_learning_rate.png")

print(f"""
ВЫВОДЫ:
  • Слишком маленькая скорость (0.0001) → сходится медленно
  • Оптимальная скорость (0.01-0.1) → сходится быстро
  • Слишком большая скорость (1.0+) → расходится

  На практике скорость обучения подбирается экспериментально.
  Для градиентного спуска на стандартизованных данных обычно
  работает диапазон 0.001 — 0.1.
""")
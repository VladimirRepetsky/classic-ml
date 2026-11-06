"""
День 2. Градиентный спуск: реализация вручную.

Сравним сырые данные и стандартизованные.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DAY_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "california_housing.csv"

print("=" * 60)
print("  ГРАДИЕНТНЫЙ СПУСК ВРУЧНУЮ")
print("=" * 60)

# Загрузка данных
df = pd.read_csv(CSV_PATH)
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# ЧАСТЬ 1: ГРАДИЕНТНЫЙ СПУСК НА СЫРЫХ ДАННЫХ
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 1: ГРАДИЕНТНЫЙ СПУСК НА СЫРЫХ ДАННЫХ")
print("=" * 60)

def gradient_descent(X, y, lr=0.01, n_iters=1000, tol=1e-8):
    """
    Градиентный спуск для линейной регрессии.
    
    Параметры:
    -----------
    X : np.ndarray, форма (n_samples, n_features)
        Матрица признаков
    y : np.ndarray, форма (n_samples,)
        Целевая переменная
    lr : float
        Скорость обучения
    n_iters : int
        Максимальное количество итераций
    tol : float
        Порог для остановки (если градиент мал)
    
    Возвращает:
    -----------
    w : np.ndarray
        Веса признаков
    b : float
        Свободный член
    history : list
        История MSE на каждой итерации
    """
    n_samples, n_features = X.shape
    
    # Инициализация весов
    w = np.zeros(n_features)
    b = 0.0
    
    history = []
    
    for i in range(n_iters):
        # Прямой проход: предсказание
        y_pred = X @ w + b
        
        # Ошибка
        error = y_pred - y
        
        # Градиенты
        #   ∂MSE/∂w = (2/n) · Xᵀ · (Xw + b - y)
        #   ∂MSE/∂b = (2/n) · Σ(Xw + b - y)
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)
        
        # Обновление весов (шаг в противоположную градиенту сторону)
        w -= lr * dw
        b -= lr * db
        
        # Метрика для отслеживания
        mse = np.mean(error ** 2)
        history.append(mse)
        
        # Проверка сходимости
        grad_norm = np.linalg.norm(dw)
        if grad_norm < tol:
            print(f"  Сошлось на итерации {i} (норма градиента: {grad_norm:.2e})")
            break
    
    return w, b, history

# Сначала посмотрим на масштаб признаков
print(f"\nМасштаб признаков (мин/макс):")
for i, name in enumerate(df.columns[:-1]):
    print(f"  {name:<12}: [{X_train[:, i].min():>10.2f}, {X_train[:, i].max():>10.2f}]")

# Попытка градиентного спуска на сырых данных
# ⚠️ Нужна очень маленькая скорость обучения!
print(f"\nГрадиентный спуск на сырых данных (нужна очень маленькая скорость):")

try:
    # Скорость обучения должна быть очень маленькой
    # потому что признаки имеют разный масштаб и большие значения
    w_raw, b_raw, history_raw = gradient_descent(
        X_train, y_train, lr=1e-7, n_iters=1000
    )
    
    # Предсказания на тесте
    y_pred_raw = X_test @ w_raw + b_raw
    mse_raw = mean_squared_error(y_test, y_pred_raw)
    final_mse = history_raw[-1]
    
    print(f"  Финальное MSE на обучающей выборке: {final_mse:.4f}")
    print(f"  MSE на тесте: {mse_raw:.4f}")
    print(f"  Bias: {b_raw:.4f}")
    print(f"  Веса: {w_raw[:4]}...")
    
except Exception as e:
    print(f"  ❌ Ошибка: {e}")

# ============================================================
# ЧАСТЬ 2: ГРАДИЕНТНЫЙ СПУСК НА СТАНДАРТИЗОВАННЫХ ДАННЫХ
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 2: ГРАДИЕНТНЫЙ СПУСК НА СТАНДАРТИЗОВАННЫХ ДАННЫХ")
print("=" * 60)

print("""
⚠️ КРИТИЧЕСКИ ВАЖНО: масштабирование и утечка данных!

Правильный порядок:
  1. Разделить данные на train и test
  2. Обучить скалер ТОЛЬКО на train
     (вычислить среднее и стандартное отклонение)
  3. Применить трансформацию к train
  4. Применить ту же трансформацию к test

НЕПРАВИЛЬНО:
  - Масштабировать все данные до разделения
  - Обучать скалер на всех данных
  - Это утечка информации из теста в модель!
""")

# ПРАВИЛЬНО: скалер обучаем только на обучающей выборке
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)        # только transform!

print(f"\nПосле стандартизации:")
print(f"  Среднее по признакам: {X_train_scaled.mean(axis=0)[:4]}")
print(f"  Стд. отклонение: {X_train_scaled.std(axis=0)[:4]}")

# Градиентный спуск на стандартизованных данных
print(f"\nГрадиентный спуск на стандартизованных данных:")

w_scaled, b_scaled, history_scaled = gradient_descent(
    X_train_scaled, y_train, lr=0.01, n_iters=1000
)

# Предсказания на тесте
y_pred_scaled = X_test_scaled @ w_scaled + b_scaled
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
final_mse_scaled = history_scaled[-1]

print(f"  Финальное MSE на обучающей выборке: {final_mse_scaled:.4f}")
print(f"  MSE на тесте: {mse_scaled:.4f}")
print(f"  Bias: {b_scaled:.4f}")
print(f"  Веса: {w_scaled[:4]}...")

# ============================================================
# ЧАСТЬ 3: СРАВНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================
print("\n" + "=" * 60)
print("  ЧАСТЬ 3: СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)

# Аналитическое решение МНК (для сравнения)
# Добавляем столбец единиц для свободного члена
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Решаем нормальное уравнение: (XᵀX)w = Xᵀy
w_ols = np.linalg.solve(X_train_b.T @ X_train_b, X_train_b.T @ y_train)

# Предсказание на тесте
y_pred_ols = X_test_b @ w_ols
mse_ols = mean_squared_error(y_test, y_pred_ols)

print(f"\n{'Метод':<35} {'MSE на тесте':>14}")
print("-" * 51)
print(f"{'МНК (аналитический)':<35} {mse_ols:>14.4f}")
print(f"{'Градиентный спуск (сырые)':<35} {mse_raw:>14.4f}")
print(f"{'Градиентный спуск (стандарт.)':<35} {mse_scaled:>14.4f}")

print(f"""
ВЫВОДЫ:
  1. Градиентный спуск на стандартизованных данных сходится
     к решению, близкому к аналитическому МНК.
  2. На сырых данных нужна очень маленькая скорость обучения,
     и сходимость медленнее.
  3. Масштабирование — критично для градиентного спуска.
""")

# ============================================================
# ЧАСТЬ 4: КРИВАЯ ОБУЧЕНИЯ
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Кривая обучения: сырые данные
axes[0].plot(history_raw, color='coral', linewidth=2)
axes[0].set_xlabel('Итерация')
axes[0].set_ylabel('MSE')
axes[0].set_title('Кривая обучения: сырые данные (нужна очень маленькая скорость)')
axes[0].grid(True, alpha=0.3)

# Кривая обучения: стандартизованные данные
axes[1].plot(history_scaled, color='steelblue', linewidth=2)
axes[1].set_xlabel('Итерация')
axes[1].set_ylabel('MSE')
axes[1].set_title('Кривая обучения: стандартизованные данные')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day02_gd_learning_curve.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day02_gd_learning_curve.png")
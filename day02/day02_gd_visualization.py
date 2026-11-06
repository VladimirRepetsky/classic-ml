"""
День 2. Траектория градиентного спуска в 2D.

Визуализация процесса оптимизации: поверхность ошибки + путь градиентного спуска.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  ТРАЕКТОРИЯ ГРАДИЕНТНОГО СПУСКА")
print("=" * 60)

# ============================================================
# ЧАСТЬ 1: СИНТЕТИЧЕСКИЕ ДАННЫЕ
# ============================================================
print("\nГенерация синтетических данных...")

np.random.seed(42)

# Простые данные: один признак
X = np.random.uniform(0, 10, 50)
y = 2.5 * X + 3 + np.random.normal(0, 2, 50)

# Стандартизация для численной устойчивости
X_mean, X_std = X.mean(), X.std()
X_scaled = (X - X_mean) / X_std

print(f"Данных: {len(X)} точек")
print(f"Истинная зависимость: y = 2.5·x + 3")
print(f"После стандартизации: среднее = {X_scaled.mean():.4f}, стд = {X_scaled.std():.4f}")

# ============================================================
# ЧАСТЬ 2: ПОВЕРХНОСТЬ ОШИБКИ
# ============================================================
print("\nВычисление поверхности ошибки...")

# Сетка весов для визуализации
w_range = np.linspace(-2, 8, 100)
b_range = np.linspace(-5, 12, 100)
W, B = np.meshgrid(w_range, b_range)
MSE_surface = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        pred = W[i, j] * X_scaled + B[i, j]
        MSE_surface[i, j] = np.mean((y - pred) ** 2)

print(f"Поверхность ошибки вычислена: {MSE_surface.shape}")
print(f"Минимальное MSE на поверхности: {MSE_surface.min():.4f}")

# ============================================================
# ЧАСТЬ 3: ГРАДИЕНТНЫЙ СПУСК С ТРАЕКТОРИЕЙ
# ============================================================
print("\nГрадиентный спуск с сохранением траектории...")

def gradient_descent_with_trajectory(X, y, lr=0.1, n_iters=50):
    """Градиентный спуск с сохранением траектории."""
    w = 0.0
    b = 0.0
    trajectory_w = [w]
    trajectory_b = [b]
    n = len(X)
    
    for i in range(n_iters):
        y_pred = w * X + b
        error = y_pred - y
        
        dw = (2 / n) * np.sum(X * error)
        db = (2 / n) * np.sum(error)
        
        w -= lr * dw
        b -= lr * db
        
        trajectory_w.append(w)
        trajectory_b.append(b)
    
    return w, b, trajectory_w, trajectory_b

# Запуск градиентного спуска
w_final, b_final, traj_w, traj_b = gradient_descent_with_trajectory(
    X_scaled, y, lr=0.1, n_iters=50
)

print(f"Финальные веса: w = {w_final:.4f}, b = {b_final:.4f}")
print(f"Истинные веса:  w = 2.5, b = 3")
print(f"Ошибка в оценке: |{w_final - 2.5:.4f}|")

# ============================================================
# ЧАСТЬ 4: ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\nВизуализация...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Данные и модель
axes[0].scatter(X_scaled, y, alpha=0.6, s=30, color='steelblue', label='Данные')
X_line = np.linspace(X_scaled.min(), X_scaled.max(), 100)
axes[0].plot(X_line, w_final * X_line + b_final, 'r-', linewidth=2,
             label=f'Модель: y = {w_final:.2f}x + {b_final:.2f}')
axes[0].plot(X_line, 2.5 * X_line + 3, 'g--', linewidth=2,
             label='Истина: y = 2.5x + 3')
axes[0].set_xlabel('X (стандартизованный)')
axes[0].set_ylabel('y')
axes[0].set_title('Линейная регрессия')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Поверхность ошибки (контурный график)
contour = axes[1].contour(W, B, MSE_surface, levels=30, cmap='viridis')
axes[1].plot(traj_w, traj_b, 'r-o', markersize=3, linewidth=1.5, label='Градиентный спуск')
axes[1].plot(w_final, b_final, 'r*', markersize=20, label='Оптимум')
axes[1].set_xlabel('w')
axes[1].set_ylabel('b')
axes[1].set_title('Поверхность ошибки + траектория')
axes[1].legend()

# 3. Поверхность ошибки (3D)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(W, B, MSE_surface, cmap='viridis', alpha=0.7, edgecolor='none')
ax3.plot(traj_w, traj_b, [np.mean((y - (w * X_scaled + b)) ** 2) for w, b in zip(traj_w, traj_b)],
         'r-', linewidth=2, label='Градиентный спуск')
ax3.set_xlabel('w')
ax3.set_ylabel('b')
ax3.set_zlabel('MSE')
ax3.set_title('3D: поверхность ошибки')
ax3.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day02_gd_trajectory.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day02_gd_trajectory.png")

print(f"""
ИНТЕРПРЕТАЦИЯ:
  • Контурный график показывает поверхность ошибки
  • Красная линия — путь градиентного спуска
  • Звёздочка — найденный оптимум
  • Алгоритм спускается по поверхности к минимуму

  На каждой итерации:
  1. Вычисляем градиент (направление наибольшего роста)
  2. Идём в противоположную сторону (к минимуму)
  3. Повторяем
""")
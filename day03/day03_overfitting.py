"""
День 3. Наглядная демонстрация переобучения и недообучения.

Обучаем модели разной сложности на МАЛОМ количестве данных
и смотрим, что происходит.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  ПЕРЕОБУЧЕНИЕ И НЕДООБУЧЕНИЕ")
print("=" * 60)

# ============================================================
# ЧАСТЬ 1: МАЛО ДАННЫХ + СЛОЖНАЯ МОДЕЛЬ = ПЕРЕОБУЧЕНИЕ
# ============================================================
print("\nГенерация данных...")

np.random.seed(42)

# Мало данных: всего 10 точек
X_small = 3 * np.random.rand(10, 1)
y_small = np.sin(X_small) + 0.5 * X_small + np.random.randn(10, 1) * 0.3

print(f"Данных: {X_small.shape[0]} точек")

# Разделение на тренинг и тест
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.3, random_state=42
)

print(f"Обучающая выборка: {X_train.shape[0]} точек")
print(f"Тестовая выборка:  {X_test.shape[0]} точек")

# ============================================================
# ЧАСТЬ 2: МОДЕЛИ РАЗНОЙ СЛОЖНОСТИ
# ============================================================
degrees = [1, 3, 9]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, degree in enumerate(degrees):
    # Полиномиальные признаки
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Обучение
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Предсказания
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    # Метрики
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    print(f"\nСтепень {degree}:")
    print(f"  MSE на тренинге: {mse_train:.4f}")
    print(f"  MSE на тесте:    {mse_test:.4f}")
    
    # Визуализация
    X_plot = np.linspace(X_small.min(), X_small.max(), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    
    axes[i].scatter(X_train, y_train, color='steelblue', s=50, label='Обучающие данные')
    axes[i].scatter(X_test, y_test, color='red', s=50, label='Тестовые данные')
    axes[i].plot(X_plot, y_plot, 'g-', linewidth=2, label='Модель')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    
    if degree == 1:
        title = f'Степень {degree} — НЕДООБУЧЕНИЕ'
    elif degree == 3:
        title = f'Степень {degree} — ХОРОШАЯ МОДЕЛЬ'
    else:
        title = f'Степень {degree} — ПЕРЕОБУЧЕНИЕ'
    
    axes[i].set_title(f'{title}\nTrain MSE={mse_train:.3f}, Test MSE={mse_test:.3f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day03_overfitting_demo.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day03_overfitting_demo.png")

# ============================================================
# ЧАСТЬ 3: ДИАГНОСТИКА
# ============================================================
print("\n" + "=" * 60)
print("  ДИАГНОСТИКА МОДЕЛЕЙ")
print("=" * 60)

print(f"""
НЕДООБУЧЕНИЕ (степень 1):
  • Модель слишком простая
  • Не улавливает нелинейную зависимость
  • Высокая ошибка И на тренинге, И на тесте
  • ЛЕЧЕНИЕ: увеличить сложность модели

ХОРОШАЯ МОДЕЛЬ (степень 3):
  • Модель адекватной сложности
  • Улавливает основную закономерность
  • Низкая ошибка и на тренинге, и на тесте
  • Небольшая разница между тренингом и тестом

ПЕРЕОБУЧЕНИЕ (степень 9):
  • Модель слишком сложная
  • Проходит через каждую точку тренировки
  • Низкая ошибка на тренинге, ВЫСОКАЯ на тесте
  • ЛЕЧЕНИЕ: уменьшить сложность, регуляризация, больше данных

ПРАВИЛО:
  Если Train MSE << Test MSE → переобучение
  Если Train MSE ≈ Test MSE, но оба высокие → недообучение
  Если Train MSE ≈ Test MSE, оба низкие → хорошая модель
""")

# ============================================================
# ЧАСТЬ 4: КАК БОРОТЬСЯ С ПЕРЕОБУЧЕНИЕМ
# ============================================================
print("=" * 60)
print("  КАК БОРОТЬСЯ С ПЕРЕОБУЧЕНИЕМ")
print("=" * 60)

print("""
1. УМЕНЬШИТЬ СЛОЖНОСТЬ МОДЕЛИ
   • Меньше степень полинома
   • Меньше признаков
   • Меньше слоёв/нейронов (для нейросетей)

2. РЕГУЛЯРИЗАЦИЯ (День 4)
   • L1 (Lasso) — зануляет неважные веса
   • L2 (Ridge) — штрафует большие веса
   • Добавляет штраф к функции потерь

3. БОЛЬШЕ ДАННЫХ
   • Собрать больше обучающих примеров
   • Аугментация данных (для изображений/текстов)

4. КРОСС-ВАЛИДАЦИЯ (День 8)
   • Более надёжная оценка качества
   • Лучший выбор гиперпараметров

5. РАННЯЯ ОСТАНОВКА (для итеративных методов)
   • Остановить обучение, когда ошибка на валидации начинает расти
""")
"""
День 3. Кривые обучения.

Кривые обучения показывают, как ошибка на тренинге и валидации
меняется с увеличением размера обучающей выборки.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  КРИВЫЕ ОБУЧЕНИЯ")
print("=" * 60)

# ============================================================
# ЧАСТЬ 1: ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================
np.random.seed(42)

X = 3 * np.random.rand(200, 1)
y = np.sin(X) + 0.5 * X + np.random.randn(200, 1) * 0.3

print(f"Данных: {X.shape[0]} точек")

# ============================================================
# ЧАСТЬ 2: КРИВЫЕ ОБУЧЕНИЯ ДЛЯ РАЗНЫХ СТЕПЕНЕЙ
# ============================================================
degrees = [1, 3, 9]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Кривые обучения
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Преобразуем отрицательный MSE в положительный
    train_errors = -train_scores.mean(axis=1)
    val_errors = -val_scores.mean(axis=1)
    
    # Визуализация
    axes[i].plot(train_sizes, train_errors, 'o-', color='steelblue', label='Ошибка на тренинге')
    axes[i].plot(train_sizes, val_errors, 'o-', color='red', label='Ошибка на валидации')
    axes[i].set_xlabel('Размер обучающей выборки')
    axes[i].set_ylabel('MSE')
    axes[i].set_title(f'Степень {degree}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'day03_learning_curves.png', dpi=150)
plt.show()

print(f"\nГрафик сохранён: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day03_learning_curves.png")

# ============================================================
# ЧАСТЬ 3: ИНТЕРПРЕТАЦИЯ
# ============================================================
print("\n" + "=" * 60)
print("  ИНТЕРПРЕТАЦИЯ КРИВЫХ ОБУЧЕНИЯ")
print("=" * 60)

print(f"""
НЕДООБУЧЕНИЕ (степень 1):
  • Ошибка на тренинге ВЫСОКАЯ
  • Ошибка на валидации ВЫСОКАЯ
  • Обе кривые сходятся на высоком уровне
  • Добавление данных НЕ помогает
  → ЛЕЧЕНИЕ: увеличить сложность модели

ХОРОШАЯ МОДЕЛЬ (степень 3):
  • Ошибка на тренинге НИЗКАЯ
  • Ошибка на валидации НИЗКАЯ
  • Небольшой разрыв между кривыми
  → Модель адекватной сложности

ПЕРЕОБУЧЕНИЕ (степень 9):
  • Ошибка на тренинге ОЧЕНЬ НИЗКАЯ
  • Ошибка на валидации ВЫСОКАЯ
  • БОЛЬШОЙ разрыв между кривыми
  • Добавление данных помогает уменьшить разрыв
  → ЛЕЧЕНИЕ: уменьшить сложность, регуляризация, больше данных

ПРАВИЛО:
  Если разрыв между кривыми большой → переобучение
  Если обе кривые на высоком уровне → недообучение
  Если кривые сходятся на низком уровне → хорошая модель
""")

# ============================================================
# ЧАСТЬ 4: РУЧНАЯ РЕАЛИЗАЦИЯ КРИВОЙ ОБУЧЕНИЯ
# ============================================================
print("\n" + "=" * 60)
print("  РУЧНАЯ РЕАЛИЗАЦИЯ КРИВОЙ ОБУЧЕНИЯ")
print("=" * 60)

def manual_learning_curve(model_factory, X, y, train_sizes):
    """
    Ручная реализация кривой обучения.
    
    Параметры:
    -----------
    model_factory : callable
        Функция, создающая новую модель
    X, y : данные
    train_sizes : list
        Размеры обучающей выборки
    
    Возвращает:
    -----------
    train_errors, val_errors : списки ошибок
    """
    train_errors = []
    val_errors = []
    
    # Разделяем на тренинг и валидацию
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    for size in train_sizes:
        if size > len(X_train):
            continue
        
        # Берём подвыборку
        X_sub = X_train[:size]
        y_sub = y_train[:size]
        
        # Создаём и обучаем модель
        model = model_factory()
        model.fit(X_sub, y_sub)
        
        # Ошибки
        train_errors.append(mean_squared_error(y_sub, model.predict(X_sub)))
        val_errors.append(mean_squared_error(y_val, model.predict(X_val)))
    
    return train_errors, val_errors

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Кривая обучения для степени 9 (переобучение)
train_sizes = [5, 10, 20, 40, 60, 80, 100, 140]
train_err, val_err = manual_learning_curve(
    lambda: make_pipeline(PolynomialFeatures(9), LinearRegression()),
    X, y, train_sizes
)

print(f"\nРучная кривая обучения (степень 9):")
print(f"{'Размер':<10} {'Train MSE':>12} {'Val MSE':>12}")
print("-" * 36)
for size, tr, va in zip(train_sizes, train_err, val_err):
    print(f"{size:<10} {tr:>12.4f} {va:>12.4f}")

print(f"""
ВЫВОД:
  С увеличением размера обучающей выборки:
  • Ошибка на тренинге растёт (модель не может запомнить всё)
  • Ошибка на валидации падает (модель обобщает лучше)
  • Разрыв уменьшается
  
  Это значит, что ПЕРЕОБУЧЕНИЕ можно исправить БОЛЬШИМ количеством данных.
""")
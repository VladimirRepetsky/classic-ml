"""
Быстрое повторение ключевых концептов Дня 1.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "california_housing.csv"

print("=" * 60)
print("  ПОВТОРЕНИЕ ДНЯ 1")
print("=" * 60)

# Загрузка данных из Дня 1
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Не найден датасет: {CSV_PATH}\n"
        f"Сначала запусти:  python day01/day01_housing.py"
    )

df = pd.read_csv(CSV_PATH)
print(f"\nДанные загружены: {df.shape[0]} строк × {df.shape[1]} столбцов")

# Разделение на признаки и целевую переменную
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values
feature_names = [c for c in df.columns if c != 'MedHouseVal']

print(f"Признаков: {X.shape[1]}")
print(f"Целевая переменная: MedHouseVal")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение модели (как в Дне 1)
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nМетрики на тесте (День 1):")
print(f"  MAE = {mae:.4f}")
print(f"  R²  = {r2:.4f}")
print(f"  Bias = {model.intercept_:.4f}")
print(f"  Веса: {model.coef_}")

print(f"\nВопросы Дня 2:")
print(f"  1. КАК модель нашла эти веса?")
print(f"  2. Что внутри model.fit()?")
print(f"  3. Почему веса такие разные по масштабу?")
print(f"\nИдём дальше.")
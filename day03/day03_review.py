"""
Быстрое повторение ключевых концептов Дня 2.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "california_housing.csv"

print("=" * 60)
print("  ПОВТОРЕНИЕ ДНЯ 2")
print("=" * 60)

# Загрузка данных из Дня 1
df = pd.read_csv(CSV_PATH)
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Стандартизация (из Дня 2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Модель
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print(f"\nЛинейная регрессия на стандартизованных данных:")
print(f"  MSE на тесте: {mse:.4f}")
print(f"  R² на тесте:  {model.score(X_test_scaled, y_test):.4f}")

print(f"""
КЛЮЧЕВЫЕ МОМЕНТЫ ДНЯ 2:
  ✅ МНК: w = (XᵀX)⁻¹Xᵀy
  ✅ Градиентный спуск: итеративная оптимизация
  ✅ Масштабирование критично для градиентного спуска
  ✅ Веса после стандартизации можно сравнивать

ВОПРОС ДНЯ 3:
  Линейная регрессия строит ПРЯМУЮ линию.
  Но что если зависимость НЕЛИНЕЙНА?
  Например, цена дома растёт с доходом,
  но после определённого уровня рост замедляется?
""")

print("Идём дальше.")
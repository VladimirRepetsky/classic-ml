"""
День 1. Линейная регрессия на California Housing.
Сохраняет датасет в общую папку data/ — он понадобится в следующих днях.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DAY_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "california_housing.csv"

print("=" * 60)
print("  ЛИНЕЙНАЯ РЕГРЕССИЯ: CALIFORNIA HOUSING")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЗКА (с кэшированием в CSV)
# ============================================================
if CSV_PATH.exists():
    print(f"\nЧитаю локальный кэш: {CSV_PATH.name}")
    df = pd.read_csv(CSV_PATH)
    feature_names = [c for c in df.columns if c != "MedHouseVal"]
else:
    print("\nСкачиваю датасет из sklearn (первый раз, нужен интернет)...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target
    feature_names = list(housing.feature_names)
    df.to_csv(CSV_PATH, index=False)
    print(f"Сохранено в: {CSV_PATH.relative_to(PROJECT_ROOT)}")

print(f"\nРазмер: {df.shape[0]} строк × {df.shape[1]} столбцов")
print(f"Признаков: {len(feature_names)}")
print(f"\nПервые 5 строк:")
print(df.head().round(3))

# ============================================================
# 2. РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA)
# ============================================================
print(f"\n--- Статистика ---")
print(df.describe().round(2))

n_missing = df.isnull().sum().sum()
print(f"\nПропущенных значений: {n_missing}")

print(f"\n--- Корреляция признаков с ценой ---")
corr = df.corr()["MedHouseVal"].drop("MedHouseVal").sort_values(ascending=False)
for name, value in corr.items():
    bar = "█" * int(abs(value) * 30)
    print(f"  {name:>12}: {value:>7.3f}  {bar}")

# ============================================================
# 3. ВИЗУАЛИЗАЦИЯ EDA
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df["MedHouseVal"], bins=50, color="steelblue", edgecolor="black", alpha=0.75)
axes[0].set_xlabel("MedHouseVal (сотни тысяч $)")
axes[0].set_ylabel("Количество районов")
axes[0].set_title("Распределение цены")

axes[1].scatter(df["MedInc"], df["MedHouseVal"], alpha=0.1, s=5, color="steelblue")
axes[1].set_xlabel("MedInc (медианный доход)")
axes[1].set_ylabel("MedHouseVal")
axes[1].set_title(f"Доход vs Цена (corr={corr['MedInc']:.3f})")

sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=axes[2], cbar_kws={"shrink": 0.8}, annot_kws={"size": 7})
axes[2].set_title("Корреляционная матрица")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "day01_housing_eda.png", dpi=150)
plt.show()

# ============================================================
# 4. X И y
# ============================================================
X = df.drop("MedHouseVal", axis=1).values
y = df["MedHouseVal"].values

print(f"\nX: {X.shape}   y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"train: {X_train.shape[0]}, test: {X_test.shape[0]}")

# ============================================================
# 5. ОБУЧЕНИЕ
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n--- Коэффициенты модели ---")
print(f"Свободный член (intercept): {model.intercept_:.4f}\n")

coef_df = pd.DataFrame({"Признак": feature_names, "Вес": model.coef_})
coef_df["|Вес|"] = coef_df["Вес"].abs()
coef_df = coef_df.sort_values("|Вес|", ascending=False)

for _, row in coef_df.iterrows():
    arrow = "↑" if row["Вес"] > 0 else "↓"
    print(f"  {arrow} {row['Признак']:>12}: {row['Вес']:>10.4f}")

print("""
ИНТЕРПРЕТАЦИЯ:
  Знак  → направление влияния (+ увеличивает цену, − уменьшает)
  |Вес| → сила влияния, НО только при сопоставимых масштабах признаков!

  ⚠️ Здесь признаки в разных единицах (доход в десятках тысяч,
     возраст в годах, население в тысячах). Сравнивать веса
     напрямую НЕКОРРЕКТНО. Масштабирование разберём в Дне 2.
""")

# ============================================================
# 6. МЕТРИКИ TRAIN vs TEST
# ============================================================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

rows = []
for name, func in [("MAE", mean_absolute_error),
                   ("MSE", mean_squared_error),
                   ("RMSE", lambda a, b: np.sqrt(mean_squared_error(a, b))),
                   ("R²", r2_score)]:
    rows.append((name, func(y_train, y_pred_train), func(y_test, y_pred_test)))

print(f"\n{'Метрика':<12} {'Train':>12} {'Test':>12} {'Разница':>12}")
print("-" * 50)
for name, tr, te in rows:
    print(f"{name:<12} {tr:>12.4f} {te:>12.4f} {te - tr:>12.4f}")

r2_te = rows[-1][2]
print(f"""
ВЫВОД:
  R²(test) = {r2_te:.4f} → модель объясняет {r2_te * 100:.1f}% вариации цен.
  Метрики train и test близки → переобучения НЕТ.
  Линейная модель слишком проста, чтобы переобучиться на 8 признаках.
""")

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_test, alpha=0.25, s=10, color="steelblue")
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
axes[0].plot(lims, lims, "r--", linewidth=2, label="Идеальное предсказание")
axes[0].set_xlabel("Реальные значения")
axes[0].set_ylabel("Предсказанные значения")
axes[0].set_title(f"Предсказание vs Реальность (R²={r2_te:.3f})")
axes[0].legend()

errors = y_test - y_pred_test
axes[1].hist(errors, bins=60, color="coral", edgecolor="black", alpha=0.75)
axes[1].axvline(0, color="black", linestyle="--", linewidth=2)
axes[1].set_xlabel("Ошибка (реальное − предсказание)")
axes[1].set_ylabel("Количество")
axes[1].set_title(f"Распределение ошибок (среднее={errors.mean():.3f})")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "day01_housing_predictions.png", dpi=150)
plt.show()

print(f"Графики сохранены в: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
print(f"Датасет для следующих дней: {CSV_PATH.relative_to(PROJECT_ROOT)}")
"""
День 1. Практика: один признак vs все признаки.
Читает датасет из общей папки data/.

ВАЖНО: сначала запусти day01_housing.py — он создаёт CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Пути ---
DAY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DAY_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "california_housing.csv"

print("=" * 60)
print("  ПРАКТИКА: 1 ПРИЗНАК vs ВСЕ ПРИЗНАКИ")
print("=" * 60)

# --- Понятная ошибка вместо трейсбека, если файла нет ---
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nНе найден датасет: {CSV_PATH}\n"
        f"Сначала запусти:  python day01/day01_housing.py"
    )

df = pd.read_csv(CSV_PATH)
y = df["MedHouseVal"].values
print(f"\nЗагружено: {df.shape[0]} строк")


def train_and_evaluate(X, label):
    """Обучает линейную регрессию и возвращает метрики + предсказания."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "label": label,
        "n_features": X.shape[1],
        "mae": mean_absolute_error(y_te, pred),
        "rmse": np.sqrt(mean_squared_error(y_te, pred)),
        "r2": r2_score(y_te, pred),
        "y_test": y_te,
        "y_pred": pred,
        "model": model,
    }


# ============================================================
# ЗАДАНИЕ 1: один признак
# ============================================================
res1 = train_and_evaluate(df[["MedInc"]].values, "1 признак (MedInc)")
print(f"\n--- {res1['label']} ---")
print(f"  w = {res1['model'].coef_[0]:.4f}, b = {res1['model'].intercept_:.4f}")
print(f"  MAE={res1['mae']:.4f}  RMSE={res1['rmse']:.4f}  R²={res1['r2']:.4f}")

# ============================================================
# ЗАДАНИЕ 2: все признаки
# ============================================================
res_all = train_and_evaluate(df.drop("MedHouseVal", axis=1).values, "Все признаки")
print(f"\n--- {res_all['label']} ({res_all['n_features']} шт.) ---")
print(f"  MAE={res_all['mae']:.4f}  RMSE={res_all['rmse']:.4f}  R²={res_all['r2']:.4f}")

# ============================================================
# ЗАДАНИЕ 3: каждый признак по отдельности
# ============================================================
print(f"\n--- Каждый признак по отдельности ---")
feature_names = [c for c in df.columns if c != "MedHouseVal"]
single_results = [train_and_evaluate(df[[f]].values, f) for f in feature_names]
single_results.sort(key=lambda r: r["r2"], reverse=True)

print(f"\n{'Признак':<14} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("-" * 46)
for r in single_results:
    print(f"{r['label']:<14} {r['mae']:>10.4f} {r['rmse']:>10.4f} {r['r2']:>10.4f}")

# ============================================================
# ЗАДАНИЕ 4: сводная таблица
# ============================================================
print(f"\n{'=' * 60}")
print("  ИТОГОВОЕ СРАВНЕНИЕ")
print("=" * 60)
print(f"\n{'Модель':<24} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("-" * 56)
print(f"{'Лучший одиночный':<24} {single_results[0]['mae']:>10.4f} "
      f"{single_results[0]['rmse']:>10.4f} {single_results[0]['r2']:>10.4f}")
print(f"{'Все 8 признаков':<24} {res_all['mae']:>10.4f} "
      f"{res_all['rmse']:>10.4f} {res_all['r2']:>10.4f}")
print(f"{'Прирост R²':<24} {'':>10} {'':>10} "
      f"{res_all['r2'] - single_results[0]['r2']:>10.4f}")

# ============================================================
# ЗАДАНИЕ 5: визуализация
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, res, color in [(axes[0], res1, "steelblue"), (axes[1], res_all, "coral")]:
    ax.scatter(res["y_test"], res["y_pred"], alpha=0.2, s=6, color=color)
    lims = [res["y_test"].min(), res["y_test"].max()]
    ax.plot(lims, lims, "r--", linewidth=2)
    ax.set_xlabel("Реальные")
    ax.set_ylabel("Предсказанные")
    ax.set_title(f"{res['label']}\nR² = {res['r2']:.3f}")

names = [r["label"] for r in single_results] + ["ВСЕ"]
r2s = [r["r2"] for r in single_results] + [res_all["r2"]]
colors = ["steelblue"] * len(single_results) + ["crimson"]
axes[2].barh(names, r2s, color=colors)
axes[2].set_xlabel("R² на тесте")
axes[2].set_title("Вклад каждого признака")
axes[2].invert_yaxis()
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "day01_practice_comparison.png", dpi=150)
plt.show()

print(f"""
ВЫВОДЫ:
  1. MedInc — самый сильный одиночный признак (R²={res1['r2']:.3f}).
  2. Все 8 признаков дают R²={res_all['r2']:.3f} — прирост есть, но не драматичный.
  3. Некоторые признаки (AveBedrms, Population) почти бесполезны в одиночку.

  Больше признаков ≠ всегда лучше. В Дне 3 увидишь, как лишние
  признаки могут УХУДШИТЬ качество на новых данных (переобучение).

График: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/day01_practice_comparison.png
""")
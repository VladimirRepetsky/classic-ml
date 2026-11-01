"""
День 1. Метрики регрессии: считаем вручную и сверяем со sklearn.
Файлов не касается — пути не нужны.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("  МЕТРИКИ РЕГРЕССИИ: РАЗБОР")
print("=" * 60)

y_true = np.array([100, 200, 300, 400, 500])
y_pred = np.array([110, 190, 320, 380, 510])
errors = y_pred - y_true

print(f"\nРеальные:    {y_true}")
print(f"Предсказано: {y_pred}")
print(f"Ошибки:      {errors}")

# ---------- MAE ----------
mae_manual = np.mean(np.abs(errors))
print(f"\n--- MAE (Mean Absolute Error) ---")
print(f"  Формула: mean(|pred - true|)")
print(f"  Вручную: {mae_manual:.4f}   sklearn: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"  Смысл: в среднем ошибаемся на {mae_manual:.1f} единиц")

# ---------- MSE ----------
mse_manual = np.mean(errors ** 2)
print(f"\n--- MSE (Mean Squared Error) ---")
print(f"  Формула: mean((pred - true)²)")
print(f"  Вручную: {mse_manual:.4f}   sklearn: {mean_squared_error(y_true, y_pred):.4f}")
print(f"  Смысл: большие ошибки штрафуются квадратично")
print(f"    ошибка 10 → штраф 100")
print(f"    ошибка 20 → штраф 400  (в 4 раза, а не в 2)")

# ---------- RMSE ----------
rmse_manual = np.sqrt(mse_manual)
print(f"\n--- RMSE (Root MSE) ---")
print(f"  Формула: √MSE")
print(f"  Вручную: {rmse_manual:.4f}   sklearn: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"  Смысл: типичная ошибка ~{rmse_manual:.1f} единиц (в единицах данных)")

# ---------- R² ----------
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_manual = 1 - ss_res / ss_tot

print(f"\n--- R² (коэффициент детерминации) ---")
print(f"  Формула: 1 - SS_res / SS_tot")
print(f"  SS_res (ошибка модели):        {ss_res:.2f}")
print(f"  SS_tot (разброс вокруг сред.): {ss_tot:.2f}")
print(f"  Вручную: {r2_manual:.6f}   sklearn: {r2_score(y_true, y_pred):.6f}")
print(f"  Смысл: объяснено {r2_manual * 100:.2f}% вариации данных")

# ---------- Сравнение моделей ----------
print(f"\n{'=' * 60}")
print("  СРАВНЕНИЕ: ИДЕАЛ / НАША / СРЕДНЕЕ / ПЛОХАЯ")
print("=" * 60)

models = {
    "Идеальная": y_true.copy(),
    "Наша": y_pred,
    "Всегда среднее": np.full_like(y_true, y_true.mean(), dtype=float),
    "Плохая": np.array([50, 350, 150, 500, 250]),
}

print(f"\n{'Модель':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("-" * 52)
for name, pred in models.items():
    print(f"{name:<20} "
          f"{mean_absolute_error(y_true, pred):>10.2f} "
          f"{np.sqrt(mean_squared_error(y_true, pred)):>10.2f} "
          f"{r2_score(y_true, pred):>10.4f}")

print("""
ОБРАТИ ВНИМАНИЕ:
  • "Всегда среднее" даёт R² = 0.0 — это точка отсчёта.
  • R² < 0 означает: модель ХУЖЕ, чем просто предсказывать среднее.
  • R² = 1.0 — идеал.

КОГДА КАКУЮ МЕТРИКУ:
  MAE  → нужна интерпретируемость, выбросы не критичны
  MSE  → большие ошибки недопустимы (штрафует сильнее)
  RMSE → то же, что MSE, но в единицах данных
  R²   → сравнение моделей между разными датасетами
""")
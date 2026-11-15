#!/usr/bin/env bash
# Запуск всех скриптов Дня 3
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  ЗАПУСК ДНЯ 3"
echo "=========================================="

for script in day03_review.py \
              day03_poly_theory.py \
              day03_poly_regression.py \
              day03_overfitting.py \
              day03_learning_curves.py \
              day03_bias_variance.py \
              day03_validation.py \
              day03_practice.py; do
    echo ""
    echo ">>> $script"
    echo "------------------------------------------"
    python "$script"
done

echo ""
echo "=========================================="
echo "  ДЕНЬ 3 ЗАВЕРШЁН"
echo "=========================================="
#!/usr/bin/env bash
# Запуск всех скриптов Дня 2
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  ЗАПУСК ДНЯ 2"
echo "=========================================="

for script in day02_review.py \
              day02_ols_theory.py \
              day02_ols_manual.py \
              day02_gradient_descent.py \
              day02_gd_visualization.py \
              day02_learning_rate.py \
              day02_scaling_interpretation.py \
              day02_practice.py; do
    echo ""
    echo ">>> $script"
    echo "------------------------------------------"
    python "$script"
done

echo ""
echo "=========================================="
echo "  ДЕНЬ 2 ЗАВЕРШЁН"
echo "=========================================="
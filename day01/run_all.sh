#!/usr/bin/env bash
# Запускает все скрипты Дня 1 по порядку.
# Использование:  bash day01/run_all.sh

set -e                                  # остановиться на первой ошибке
cd "$(dirname "$0")"                    # перейти в папку скрипта

echo "=========================================="
echo "  ЗАПУСК ДНЯ 1"
echo "=========================================="

for script in day01_ml_intro.py \
              day01_linear_regression.py \
              day01_metrics.py \
              day01_housing.py \
              day01_practice.py; do
    echo ""
    echo ">>> $script"
    echo "------------------------------------------"
    python "$script"
done

echo ""
echo "=========================================="
echo "  ДЕНЬ 1 ЗАВЕРШЁН"
echo "=========================================="
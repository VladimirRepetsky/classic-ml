# Classic ML — Блок 2

Классическое машинное обучение с scikit-learn.
Продолжение репозитория [ml-fundamentals](https://github.com/VladimirRepetsky/ml-fundamentals) (Блок 1: основы программирования, 19 дней).

## Структура

```text
classic-ml/
├── data/          общие датасеты для всех дней
├── day01/         один день = одна папка
│   ├── *.py       скрипты дня
│   ├── README.md  конспект дня
│   ├── run_all.sh запуск всего дня
│   └── outputs/   графики дня
├── day02/
└── ...
```

## Дни

| День | Тема | Папка |
|---|---|---|
| 1 | Введение в ML. Линейная регрессия. Метрики | [day01](day01/) |
| 2 | МНК, градиентный спуск, масштабирование | [day02](day02/) |
| 3 | Полиномиальная регрессия, переобучение | [day03](day03/) |
| 4 | Регуляризация: Ridge, Lasso | *скоро* |
| 5 | Логистическая регрессия | *скоро* |
| 6 | KNN и деревья решений | *скоро* |
| 7 | Ансамбли: Random Forest, Gradient Boosting | *скоро* |
| 8 | Кросс-валидация и подбор гиперпараметров | *скоро* |
| 9 | Кластеризация и PCA | *скоро* |
| 10 | Сквозной проект блока | *скоро* |

## Установка

```bash
git clone https://github.com/VladimirRepetsky/classic-ml.git
cd classic-ml
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
pip install -r requirements.txt
```

## Запуск дня

```bash
bash day01/run_all.sh
```

## Стек

`numpy` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`
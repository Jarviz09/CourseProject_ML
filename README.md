# Прогнозирование потребления электроэнергии на основе временных рядов

Курсовой проект посвящен разработке нейронной сети для прогнозирования потребления электроэнергии на следующий час по данным за предыдущие 48 часов.

## Идея проекта

В проекте решается задача регрессии временного ряда. Модель получает последовательность признаков за последние 48 часов и прогнозирует значение `Global_active_power` на следующий час.

Используемые признаки:

- `Global_active_power` – активное потребление электроэнергии;
- `temperature` – температура воздуха;
- `humidity` – влажность;
- `hour_sin`, `hour_cos` – циклическое представление часа суток;
- `is_weekend` – признак выходного дня.

## Структура проекта

```text
energy_forecasting_project/
├── configs/
│   └── config.yaml
├── data/
│   └── energy_consumption.csv
├── outputs/
│   ├── course_presentation.pptx
│   ├── training_loss.png
│   ├── validation_rmse.png
│   ├── prediction_plot.png
│   ├── actual_vs_predicted.png
│   ├── daily_profile.png
│   ├── metrics.json
│   └── speaker_notes.md
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── README.md
└── requirements.txt
```

## Как запустить

Установить зависимости:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Сгенерировать данные, графики и демонстрационные результаты:

```bash
python scripts/generate_outputs.py
```

Обучить LSTM-модель:

```bash
python -m src.train
```

Оценить качество модели:

```bash
python -m src.evaluate
```

## Метод решения

Для прогнозирования используется LSTM-сеть. Такая архитектура подходит для временных рядов, так как учитывает порядок наблюдений и может использовать информацию из предыдущих временных шагов.


## Метрики качества

Для оценки используются метрики регрессии:

- **MAE** – средняя абсолютная ошибка;
- **RMSE** – корень из среднеквадратичной ошибки;
- **MAPE** – средняя процентная ошибка;
- **R2** – доля объясненной дисперсии.

Итоговые значения сохраняются в `outputs/metrics.json`.
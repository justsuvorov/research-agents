# Спецификация MLAgent (SDD)

## 1. Цель
Обучить GLM модель на датасете и сохранить результаты.

## 2. Входные данные
- **dataset.csv** — датасет от DataAgent
- **agent_config.yaml** — конфиг (target_variable, features, hyperparameters)

## 3. Ограничения (GUARDRAILS)

### 3.1 ModelRunner (outboxml)
- Max rows в датасете: 100 000
- Max features: 100 (обрезать если больше)
- Min rows для обучения: 10
- Target variable: обязательно существует в датасете
- Таймаут обучения: 10 минут
- Если обучение зависает > 10 мин: kill процесс и RuntimeError

### 3.2 FigurePlotter
- Max фигур: 10
- Таймаут на фигуру: 5 сек
- Если фигура зависает: skip и continue

### 3.3 ResultExporter
- Max coefficients: 100
- Max metrics: 50
- Валидация JSON структуры перед сохранением
- Если экспорт fails: raise RuntimeError (не скрывать)

### 3.4 Общие
- Таймаут агента: 15 минут
- Контроль памяти: если > 2GB → warning и завершение

## 4. Выходные данные
- model_results.json (коэффициенты, метрики)
- figures/ (диагностические графики)

## 5. Критерии успеха
✓ model_results.json создан и валиден
✓ Минимум 5 коэффициентов
✓ Метрики содержат R², RMSE, AIC
✓ Фигуры созданы (если format=pdf/png)

# Спецификация DataAgent (SDD)

## 1. Цель
Собрать датасет из трех источников: paper extraction, standards calculations, user data.

## 2. Входные данные
- **papers.json** — релевантные статьи от ResearchAgent
- **agent_config.yaml** — конфиг (extraction_rules, calculations, engineering_calculations, user_data)
- **prompts/data/** — пользовательские промпты (system.txt, paper_extractor.txt, standards_calculator.txt, engineering_calculator.txt)

## 3. Ограничения (GUARDRAILS)

### 3.1 PaperExtractor
- Max статей: max_papers (обычно 30)
- Таймаут на статью: 20 сек
- Max tokens: 2048
- Попытки парсинга: 1
- Fallback: пропустить статью если ошибка
- Max rows на статью: 10 (не лезть в бесконечность)

### 3.2 StandardsCalculator
- Max комбинаций: product(parameter_ranges) ограничить до 1000
- Таймаут на batch: 30 сек
- Max tokens: 4096
- Если комбинаций > 1000: предупреждение и обрезка

### 3.3 EngineeringCalculator
- Max комбинаций на rule: 500
- Batch size: 20 параметров за раз
- Таймаут на batch: 40 сек
- Max tokens: 8192
- Retry API: 3 раза
- Если ошибок > 50%: skip весь rule

### 3.4 DatasetAssembler
- Min rows: 10 (иначе RuntimeError)
- Max rows: 100 000 (обрезать если больше)
- Проверка дублей: drop по (source, source_type, key_fields)
- Валидация типов перед сохранением

### 3.5 Общие
- Таймаут агента: 30 минут
- Контроль памяти: логировать размер на каждом этапе

## 4. Выходные данные
- dataset.csv (минимум 10 строк)
- dataset_metadata.json (описание колонок)

## 5. Критерии успеха
✓ dataset.csv имеет >= 10 rows
✓ Все колонки валидны
✓ metadata.json содержит описание
✓ Нет дублей

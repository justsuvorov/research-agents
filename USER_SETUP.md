# Инструкция для пользователя: Настройка и запуск

## Шаг 1: Подготовка окружения

### 1.1 Установите зависимости
```bash
pip install -r requirements.txt
```

### 1.2 Скопируйте .env шаблон и заполните ключи
```bash
cp .env.example .env
```

**Заполните в `.env`:**
```
GEMINI_API_KEY=<ваш ключ от Google Gemini>
AI_MODEL_NAME=gemini-2.0-flash

# Опционально (если используете Semantic Scholar)
SEMANTIC_SCHOLAR_API_KEY=<ваш ключ>
MAILTO=your@email.com

# Опционально (если используете eLIBRARY)
ELIBRARY_API_TOKEN=<ваш ключ>
```

---

## Шаг 2: Создайте вашу исследовательскую цель

### 2.1 Создайте файл `research_goal.txt`
```bash
# Пример content:
Разработка математической модели для оценки интенсивности изнашивания 
зубчатой передачи опорно-поворотного устройства (ОПУ) судовых кранов 
с учётом динамических нагрузок...
```

**Требования к research_goal.txt:**
- ✓ Минимум 50 слов
- ✓ Ясное описание предметной области
- ✓ Указание ключевых факторов/параметров
- ✓ На русском или английском

---

## Шаг 3: Настройте системные промпты (один раз)

### 3.1 Системный промпт
**Файл:** `prompts/research/system.txt`

**Содержание:**
```
Ты опытный научный исследователь, специалист в инженерии и материаловедении.
Твоя задача: анализировать научные статьи, выделять ключевую информацию, 
определять релевантность к исследовательской цели.

Ответы давай на русском языке.
Будь точен в деталях и формулировках.
```

### 3.2 Промпт QueryBuilder
**Файл:** `prompts/research/query_builder.txt`

**Содержание:**
```
Исследовательская цель: {goal}

Сгенерируй {n_queries} поисковых запросов на английском языке для поиска 
научных статей по этой теме. Запросы должны быть специфичными и охватывать 
разные аспекты исследования.

Возвращай результат как JSON массив строк:
["запрос 1", "запрос 2", ...]
```

### 3.3 Промпт PaperAnalyzer
**Файл:** `prompts/research/paper_analyzer.txt`

**Содержание:**
```
Проанализируй научную статью и определи её релевантность для исследования 
математической модели износа зубчатых передач.

Статья: {title}
Авторы: {authors}
Год: {year}
Абстракт: {abstract}

Верни JSON объект:
{{
  "passes_domain_filter": true/false,
  "relevance_score": 0.0-1.0,
  "category": "theory|experiments|standards|tools|other",
  "summary": "краткое резюме",
  "key_equation": "главное уравнение если есть",
  "gap_analysis": "пробелы в исследовании"
}}
```

### 3.4 Промпт Synthesizer
**Файл:** `prompts/research/synthesizer.txt`

**Содержание:**
```
Напиши раздел литературного обзора по категории: {category}

Исследовательская цель: {goal}

Используй следующие источники:
{sources_block}

Раздел должен:
- Быть написан на русском
- Содержать 300-500 слов
- Объединять информацию из разных статей
- Показывать связи и противоречия
- Выявлять пробелы в знаниях

Начни прямо с содержания, без заголовков.
```

---

## Шаг 4: Настройте конфиг для вашего исследования

### 4.1 Обновите `agent_config.yaml`

**research секция:**
```yaml
research:
  sources: [arxiv, semantic_scholar, mdpi]  # Какие источники использовать
  max_papers: 25                             # Сколько статей анализировать
  citation_format: APA                       # Формат цитирования
  language: en                               # Язык поиска
```

**data секция:** (оставьте пусто для первого запуска)
```yaml
data:
  output_format: csv
  extraction_rules: []
  calculations: []
  engineering_calculations: []
  user_data: null
```

**ml секция:** (оставьте пусто для первого запуска)
```yaml
ml:
  model: GLM
  target_variable: null
  features: []
  hyperparameters: {}
```

---

## Шаг 5: Запустите пайплайн

### 5.1 Запуск ResearchAgent (первый запуск)
```bash
python main.py \
  --goal research_goal.txt \
  --config agent_config.yaml \
  --output ./output
```

**Ожидаемое время:** 5-10 минут

### 5.2 Проверьте результаты
```bash
ls -la output/
# Должны быть:
# - literature_review.md
# - references.bib
# - papers.json
# - run_context.json
```

---

## Шаг 6: Следующие шаги (после ResearchAgent)

### 6.1 DataAgent (сбор данных)
Заполните в `agent_config.yaml`:
```yaml
data:
  extraction_rules:
    - name: "Параметр 1"
      type: numeric
      description: "Описание"
      unit: "ед.изм"
```

Создайте промпты:
- `prompts/data/system.txt`
- `prompts/data/paper_extractor.txt`
- `prompts/data/standards_calculator.txt`
- `prompts/data/engineering_calculator.txt`

### 6.2 MLAgent (обучение модели)
Заполните в `agent_config.yaml`:
```yaml
ml:
  target_variable: "название колонки для предсказания"
  features: ["колонка1", "колонка2"]  # или оставьте пусто для всех
```

### 6.3 ReportAgent (отчет)
Создайте промпт:
- `prompts/report/review.txt`

---

## Советы по оптимизации

### Для экономии токенов:
- Используйте max_papers: 15-20 (вместо 30)
- Сужайте исследовательскую цель (более специфичная → меньше нерелевантных статей)
- Используйте одну категорию papers для первого теста

### Для лучших результатов:
- Детально опишите research_goal.txt (ключевые параметры, стандарты)
- Кастомизируйте системный промпт под вашу предметную область
- Проверьте источники (доступны ли они? Много ли по вашей теме?)

### Если ошибка "Insufficient relevant sources":
- Расширьте research_goal.txt
- Добавьте источники (arxiv, mdpi, elibrary)
- Увеличьте max_papers

---

## Структура проекта

```
research-agents/
├── research_goal.txt           ← ВЫ создаете (описание цели)
├── agent_config.yaml           ← ВЫ редактируете (конфиг)
├── .env                        ← ВЫ заполняете (API ключи)
├── prompts/
│   ├── research/
│   │   ├── system.txt         ← ВЫ создаете (системный промпт)
│   │   ├── query_builder.txt  ← ВЫ создаете
│   │   ├── paper_analyzer.txt ← ВЫ создаете
│   │   └── synthesizer.txt    ← ВЫ создаете
│   └── data/
│       └── ... (для следующих запусков)
├── specs/
│   ├── 01_research_agent.spec.md
│   ├── 02_data_agent.spec.md
│   └── ... (справочные файлы)
├── output/
│   ├── literature_review.md
│   ├── references.bib
│   ├── papers.json
│   └── run_context.json
└── main.py
```

---

## Контроль качества

**После запуска ResearchAgent проверьте:**

```bash
# 1. Файлы созданы?
ls -la output/

# 2. JSON валиден?
python -m json.tool output/papers.json > /dev/null

# 3. BibTeX валиден?
grep -c "^@" output/references.bib  # должно быть >= 5

# 4. Обзор содержательный?
wc -w output/literature_review.md   # должно быть >= 500 слов
```

---

## FAQ

**Q: Как изменить температуру LLM?**
A: Отредактируйте в `.env`:
```
GEN_TEMPERATURE_CREATIVE=0.8
GEN_TEMPERATURE_ANALYTICAL=0.5
```

**Q: Что если статьи не найдены?**
A: Проверьте:
1. Интернет соединение
2. API ключи в .env
3. research_goal.txt (слишком узкая цель?)

**Q: Как использовать eLIBRARY?**
A: Получите токен на https://elibrary.ru/projects/api, добавьте в .env

**Q: Сколько токенов использует ResearchAgent?**
A: ~40-50K на полный запуск с 25 статьями

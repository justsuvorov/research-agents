# Спецификация ResearchAgent (SDD)

## 1. Цель
Поиск и анализ научных статей по исследовательской цели. Создание литературного обзора.

## 2. Входные данные
- **research_goal.txt** — текст цели исследования
- **agent_config.yaml** — конфиг (sources, max_papers, language)
- **prompts/** — пользовательские промпты (system.txt, query_builder.txt, paper_analyzer.txt, synthesizer.txt)

## 3. Ограничения (GUARDRAILS)

### 3.1 QueryBuilder
- Таймаут: 30 сек
- Max tokens: 512
- Fallback: 6 встроенных запросов если JSON невалиден
- Попытки: 1

### 3.2 Searchers
- Max на источник: max_results (обычно 4)
- Таймаут на источник: 15 сек
- Таймаут всего: 60 сек
- Если недоступен: skip → continue (не зацикливаться)

### 3.3 PaperAnalyzer
- Max статей: max_papers
- Таймаут на статью: 20 сек
- Max tokens: 1024
- Если парс невалиден: skip статья

### 3.4 Фильтрация
- Минимум релевантных: 5
- Если < 5: RuntimeError (не зацикливаться в поиске)

### 3.5 Synthesizer
- Max разделов: 5
- Таймаут на раздел: 30 сек
- Max tokens: 2048

### 3.6 Общие
- Общий таймаут: 15 минут
- Retry сетевых ошибок: 2 раза
- Retry rate-limit: 3 раза
- Логирование токенов на каждом этапе

## 4. Выходные данные
- literature_review.md (Markdown, минимум 500 символов)
- references.bib (BibTeX, минимум 5 записей)
- papers.json (валидный JSON)

## 5. Критерии успеха
✓ Found >= 5 relevant papers
✓ All files created and non-empty
✓ JSON valid
✓ BibTeX entries > 0

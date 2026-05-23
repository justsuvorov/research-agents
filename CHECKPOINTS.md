# Механизм контрольных точек (Checkpoints)

## Цель
Сохранять промежуточные результаты после каждого шага, чтобы при прерывании запуск продолжился с последней точки без повтора работы и трат токенов.

---

## 1. Архитектура контрольных точек

### 1.1 Структура сохранения

```
output/
├── run_context.json              ← Основное состояние
└── checkpoints/
    ├── research/
    │   ├── queries.json          ← Сгенерированные запросы
    │   ├── papers_raw.json       ← Найденные статьи (до дедупа)
    │   ├── papers_dedup.json     ← После дедупликации
    │   ├── analyses.json         ← Анализы всех статей
    │   ├── analyses_relevant.json ← Только релевантные
    │   └── synthesis.json        ← Синтезированные разделы
    ├── data/
    │   ├── paper_rows.json       ← Извлеченные из статей
    │   ├── calc_rows.json        ← Из расчетов
    │   ├── eng_calc_rows.json    ← Из инженерных расчетов
    │   └── user_rows.json        ← Из пользовательских данных
    ├── ml/
    │   ├── dataset_clean.csv     ← Очищенный датасет
    │   ├── model_trained.pkl     ← Обученная модель
    │   └── figures_list.json     ← Список созданных фигур
    └── report/
        └── draft.md              ← Черновик отчета
```

### 1.2 Уровни сохранения

**Уровень 1: Критичные артефакты** (обязательны)
```python
# Сохраняются после каждого агента
run_context.json         # Полное состояние
checkpoints/*/main.json  # Основной результат
```

**Уровень 2: Промежуточные данные** (для восстановления)
```python
# Сохраняются внутри агента после ключевых этапов
checkpoints/research/queries.json
checkpoints/research/papers_raw.json
checkpoints/data/paper_rows.json
```

**Уровень 3: Debug информация** (опционально)
```python
# Сохраняются для анализа ошибок
checkpoints/research/debug_log.json  # Статистика
```

---

## 2. Структура RunContext для checkpoints

```python
class AgentCheckpoint(BaseModel):
    """Контрольная точка агента"""
    agent_name: str              # "research", "data", "ml", "report"
    status: AgentStatus          # COMPLETED, RUNNING, FAILED
    timestamp: str               # ISO 8601
    checkpoint_dir: str          # путь к checkpoints/{agent}/
    
    # Метаданные для восстановления
    step_completed: int          # Сколько итемов обработано
    total_items: int             # Всего нужно обработать
    tokens_used: int             # Токены на данный момент
    
    # Файлы контрольной точки
    main_artifact: str           # Основной результат
    intermediate_artifacts: dict # {name: path}
    
    errors: list[str]            # Ошибки на этапе
    recovery_possible: bool      # Можно ли восстановиться


class RunContext(BaseModel):
    # ... существующие поля ...
    
    checkpoints: dict[str, AgentCheckpoint] = {}  # Добавить
    
    def checkpoint(
        self, 
        agent_name: str,
        step: int,
        total: int,
        artifacts: dict[str, str],
        status: AgentStatus
    ) -> None:
        """Сохранить контрольную точку"""
        cp = AgentCheckpoint(
            agent_name=agent_name,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checkpoint_dir=f"{self.output_dir}/checkpoints/{agent_name}",
            step_completed=step,
            total_items=total,
            tokens_used=self.tokens_used,
            intermediate_artifacts=artifacts,
            recovery_possible=True,
        )
        self.checkpoints[agent_name] = cp
        self.save()
    
    def last_completed(self) -> str | None:
        """Какой агент последний завершил работу?"""
        agents = ["research", "data", "ml", "report"]
        for agent in agents:
            if agent in self.checkpoints:
                cp = self.checkpoints[agent]
                if cp.status == AgentStatus.COMPLETED:
                    return agent
        return None
    
    def can_resume_from(self, agent_name: str) -> bool:
        """Можно ли продолжить с этого агента?"""
        if agent_name not in self.checkpoints:
            return False
        cp = self.checkpoints[agent_name]
        return cp.recovery_possible and cp.status in [
            AgentStatus.COMPLETED,
            AgentStatus.FAILED
        ]
```

---

## 3. Реализация для каждого агента

### 3.1 ResearchAgent

```python
class ResearchAgent(BaseAgent):
    
    def run(self):
        # Проверка восстановления
        if self._can_resume():
            logger.info("[ResearchAgent] Resuming from checkpoint")
            self._resume_from_checkpoint()
            return
        
        cfg = ResearchConfig(**self.ctx.config["research"])
        
        # Этап 1: Генерация запросов
        queries = self._query_builder.search_queries(self.ctx.goal, n_queries=6)
        self.ctx.checkpoint("research", 1, 6, {"queries": queries_path}, RUNNING)
        
        # Этап 2: Поиск статей
        papers = self._search_all(queries, cfg)
        self._save_checkpoint(f"{checkpoint_dir}/papers_raw.json", papers)
        self.ctx.checkpoint("research", 2, 6, {"papers_raw": path}, RUNNING)
        
        # Этап 3: Дедупликация
        papers = self._deduplicated(papers)
        self._save_checkpoint(f"{checkpoint_dir}/papers_dedup.json", papers)
        self.ctx.checkpoint("research", 3, 6, {"papers_dedup": path}, RUNNING)
        
        # Этап 4: Анализ
        report = LiteratureReport(goal=self.ctx.goal)
        for i, paper in enumerate(papers[: cfg.max_papers]):
            analysis = self._paper_analyzer.paper_analysis(paper)
            report.analyses.append(analysis)
            
            # Сохранять промежуточный результат каждые 5 статей
            if (i + 1) % 5 == 0:
                self._save_checkpoint(f"{checkpoint_dir}/analyses.json", report)
                self.ctx.checkpoint("research", 4, 6, {"analyses": path}, RUNNING)
                logger.info("[ResearchAgent] Progress: {}/{}", i+1, len(papers))
        
        relevant = report.relevant()
        self._save_checkpoint(f"{checkpoint_dir}/analyses_relevant.json", relevant)
        self.ctx.checkpoint("research", 5, 6, {"relevant": path}, RUNNING)
        
        # Этап 5: Синтез
        sections = self._synthesizer.literature_review_sections(report)
        self._save_checkpoint(f"{checkpoint_dir}/synthesis.json", sections)
        
        # Этап 6: Экспорт
        self._export_results(report, sections)
        self.ctx.checkpoint("research", 6, 6, {"final": artifacts}, COMPLETED)
    
    def _can_resume(self) -> bool:
        """Можно ли возобновить с этого агента?"""
        cp = self.ctx.checkpoints.get("research")
        if not cp:
            return False
        return cp.recovery_possible and cp.step_completed > 0
    
    def _resume_from_checkpoint(self):
        """Восстановиться из контрольной точки"""
        cp = self.ctx.checkpoints["research"]
        checkpoint_dir = cp.checkpoint_dir
        
        # Определить, с какого этапа начать
        if cp.step_completed >= 5:
            # Уже есть анализы → только синтез и экспорт
            analyses_path = f"{checkpoint_dir}/analyses_relevant.json"
            analyses = self._load_json(analyses_path)
            report = LiteratureReport.from_analyses(analyses)
        else:
            # Перезагрузить все анализы и продолжить
            analyses_path = f"{checkpoint_dir}/analyses.json"
            analyses = self._load_json(analyses_path)
            report = LiteratureReport.from_analyses(analyses)
        
        # Выполнить оставшиеся этапы
        sections = self._synthesizer.literature_review_sections(report)
        self._export_results(report, sections)
        self.ctx.checkpoint("research", 6, 6, {"final": artifacts}, COMPLETED)
```

### 3.2 DataAgent (аналогично)

```python
class DataAgent(BaseAgent):
    
    def run(self):
        if self._can_resume():
            self._resume_from_checkpoint()
            return
        
        cfg = DataConfig(**self.ctx.config["data"])
        checkpoint_dir = f"{self.ctx.output_dir}/checkpoints/data"
        
        # Этап 1: Paper extraction
        paper_rows = self._paper_rows(cfg)
        self._save_checkpoint(f"{checkpoint_dir}/paper_rows.json", paper_rows)
        self.ctx.checkpoint("data", 1, 4, {"paper_rows": path}, RUNNING)
        
        # Этап 2: Standards calculations
        calc_rows = self._calculation_rows(cfg)
        self._save_checkpoint(f"{checkpoint_dir}/calc_rows.json", calc_rows)
        self.ctx.checkpoint("data", 2, 4, {"calc_rows": path}, RUNNING)
        
        # Этап 3: Engineering calculations
        eng_rows = self._engineering_rows(cfg)
        self._save_checkpoint(f"{checkpoint_dir}/eng_rows.json", eng_rows)
        self.ctx.checkpoint("data", 3, 4, {"eng_rows": path}, RUNNING)
        
        # Этап 4: User data
        user_rows = self._user_rows(cfg)
        self._save_checkpoint(f"{checkpoint_dir}/user_rows.json", user_rows)
        
        # Сборка датасета
        dataset_path = self._assemble_dataset(paper_rows, calc_rows, eng_rows, user_rows)
        self.ctx.checkpoint("data", 4, 4, {"dataset": dataset_path}, COMPLETED)
    
    def _resume_from_checkpoint(self):
        """Восстановиться и продолжить"""
        cp = self.ctx.checkpoints["data"]
        checkpoint_dir = cp.checkpoint_dir
        
        # Загрузить уже собранные данные
        rows = []
        for step in range(1, cp.step_completed + 1):
            file_map = {
                1: "paper_rows.json",
                2: "calc_rows.json",
                3: "eng_rows.json",
                4: "user_rows.json",
            }
            path = f"{checkpoint_dir}/{file_map[step]}"
            rows.extend(self._load_json(path))
        
        # Пересчитать оставшиеся (если есть)
        # ...
        
        # Собрать датасет
        dataset_path = self._assemble_dataset(*rows)
        self.ctx.checkpoint("data", 4, 4, {"dataset": dataset_path}, COMPLETED)
```

---

## 4. Логика пайплайна с checkpoints

```python
class ResearchPipeline:
    
    def result(self) -> RunContext:
        """Запустить с поддержкой восстановления"""
        logger.info("Pipeline started run_id={}", self.ctx.run_id)
        
        # Определить, откуда начать
        last_completed = self.ctx.last_completed()
        start_idx = 0
        
        if last_completed:
            agents_order = ["research", "data", "ml", "report"]
            start_idx = agents_order.index(last_completed) + 1
            logger.info("Resuming from: {} (last completed)", last_completed)
        
        for idx, (name, agent) in enumerate(self._agents):
            if idx < start_idx:
                logger.debug("Skipping {}: already completed", name)
                continue
            
            logger.info("Starting agent: {}", name)
            
            # Запустить агент (он проверит checkpoints сам)
            agent.execute()
            
            if self.ctx.errors.get(name):
                logger.error("Agent failed: {} — not continuing", name)
                break
        
        logger.info("Pipeline finished run_id={}", self.ctx.run_id)
        return self.ctx
```

---

## 5. Команды для пользователя

### Запуск с нуля
```bash
python main.py --goal research_goal.txt --config agent_config.yaml --output ./output
```

### Возобновление после прерывания
```bash
# Автоматическое — просто запустить снова!
python main.py --goal research_goal.txt --config agent_config.yaml --output ./output
# Пайплайн проверит checkpoints и продолжит
```

### Полная переустановка (с нуля)
```bash
python main.py --goal research_goal.txt --config agent_config.yaml --output ./output --reset
# Удалит все checkpoints и начнет заново
```

### Статус checkpoints
```bash
python -m research_agents.tools.checkpoint_status --output ./output
# Покажет:
# research: COMPLETED (4/6 steps, queries + papers + analyses)
# data: RUNNING (2/4 steps, paper_rows collected)
# ml: PENDING
# report: PENDING
```

---

## 6. Структура checkpoint файла

**checkpoints/research/queries.json:**
```json
[
  "slewing bearing wear boundary lubrication",
  "composite boom crane weight reduction",
  ...
]
```

**checkpoints/research/papers_raw.json:**
```json
{
  "papers": [
    {
      "title": "...",
      "doi": "...",
      "source": "arxiv",
      "abstract": "...",
      "authors": [...],
      "year": 2024
    }
  ],
  "search_queries": [...],
  "timestamp": "2026-05-23T12:30:00Z"
}
```

**checkpoints/research/analyses_relevant.json:**
```json
{
  "analyses": [
    {
      "paper": {...},
      "passes_domain_filter": true,
      "relevance_score": 0.85,
      "category": "experimental",
      "summary": "..."
    }
  ],
  "count": 12,
  "timestamp": "2026-05-23T12:45:00Z"
}
```

---

## 7. Правила сохранения

### Когда сохранять?

✅ **После каждого агента завершит работу**
```python
agent.execute()
ctx.checkpoint(agent_name, COMPLETED)
```

✅ **Внутри агента после ключевых этапов**
```python
# Каждые N итемов (например, каждые 5 статей)
for i, item in enumerate(items):
    process(item)
    if (i + 1) % 5 == 0:
        ctx.checkpoint(agent_name, step, RUNNING)
```

✅ **После критичных операций**
```python
papers = deduplicate(papers)
save_checkpoint(papers)  # На случай если следующий этап упадет
```

### Когда НЕ сохранять?

❌ **После каждой строки/статьи** — слишком частое I/O
❌ **Только после успеха** — нужно сохранять и FAILED состояние для анализа

---

## 8. Очистка и удаление checkpoints

### Автоматическая очистка (опционально)
```yaml
# agent_config.yaml
cleanup:
  keep_last_n_checkpoints: 3  # Хранить только последние 3
  archive_old: true           # Архивировать старые
```

### Ручная очистка
```bash
# Удалить все checkpoints для research агента
rm -rf output/checkpoints/research

# Удалить все checkpoints
rm -rf output/checkpoints

# Архивировать checkpoints
tar -czf output/checkpoints.tar.gz output/checkpoints
```

---

## 9. Мониторинг и диагностика

### Файл статистики
**checkpoints/.stats.json:**
```json
{
  "research": {
    "completed_steps": 4,
    "total_steps": 6,
    "tokens_used": 25000,
    "estimated_remaining_tokens": 15000,
    "time_elapsed": "00:15:30",
    "estimated_total_time": "00:22:45"
  },
  "data": {
    "completed_steps": 0,
    "total_steps": 4
  }
}
```

### Восстановление после сбоя
```bash
# Если падает с ошибкой — checkpoints остаются
# На следующий запуск они автоматически используются

python main.py --goal research_goal.txt ... 
# Продолжит с checkpoint

# Если нужно откатиться на один агент назад:
python main.py --goal research_goal.txt ... --resume-from data
```

---

## 10. Примеры сценариев

### Сценарий 1: Нормальное выполнение
```
Запуск 1: research → COMPLETED (6 мин, 25K токенов)
          data → RUNNING (прерывается на 50%)
Запуск 2: data → RUNNING (с checkpoint) → COMPLETED (1 мин, 5K токенов)
          ml → COMPLETED (5 мин)
          report → COMPLETED (2 мин)
```

### Сценарий 2: Многократное прерывание
```
Запуск 1: research (5/6 статей) → прерывается
Запуск 2: research (загрузил 5, обработал 6) → data (1/4) → прерывается
Запуск 3: data (2/4) → ml → report
# Без дополнительных трат токенов на research!
```

### Сценарий 3: Исправление конфига
```
Запуск 1: research → COMPLETED
          data (extraction_rules были пусты) → COMPLETED
Пользователь: добавил extraction_rules в config
Запуск 2: python main.py ... --reset-from data
          # Checkpoints research остались, data переделана с новым конфигом
```

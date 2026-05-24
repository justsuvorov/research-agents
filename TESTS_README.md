# Тесты для research-agents

Полный набор unit и интеграционных тестов для пайплайна.

## Структура тестов

```
tests/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── test_research_agent.py      # Integration tests для ResearchAgent
│   └── test_ml_agent.py             # Integration tests для MLAgent
├── test_pydantic_models.py          # Unit tests для RunContext, AgentCheckpoint
├── test_config.py                   # Unit tests для конфига
├── test_base_agent.py               # Unit tests для BaseAgent
└── test_checkpoint_recovery.py      # Integration tests для checkpoints
```

## Какие тесты включены

### 1. Unit Tests (быстрые, без API)

#### `test_pydantic_models.py` (30+ тестов)
- ✅ AgentCheckpoint creation and serialization
- ✅ RunContext save/load
- ✅ Checkpoint methods (checkpoint, checkpoint_completed, checkpoint_failed)
- ✅ Resume logic (last_completed_agent, can_resume)
- ✅ Artifact management

#### `test_config.py` (15+ тестов)
- ✅ research_goal() loading
- ✅ ResearchConfig validation
- ✅ DataConfig validation
- ✅ MLConfig validation
- ✅ AgentConfig loading from YAML

#### `test_base_agent.py` (10+ тестов)
- ✅ Agent execution
- ✅ Idempotency on second execute()
- ✅ Status transitions (PENDING → RUNNING → COMPLETED)
- ✅ Error handling and FAILED status
- ✅ Existing RunContext loading

### 2. Integration Tests (тестируют реальные компоненты)

#### `test_research_agent.py` (15+ тестов)
- ✅ Full ResearchAgent execution with mocked LLM
- ✅ Artifact generation (literature_review.md, references.bib, papers.json)
- ✅ Checkpoint creation (queries.json, papers_raw.json, etc.)
- ✅ Idempotency (second execute() is skipped)
- ✅ Minimum papers requirement (>= 5 relevant papers)
- ✅ BibTeX format validation
- ✅ JSON format validation

#### `test_checkpoint_recovery.py` (15+ тестов)
- ✅ Checkpoint saved after each step
- ✅ All checkpoint files created
- ✅ Checkpoint persistence across load/reload
- ✅ Second execution skips (no duplicate API calls)
- ✅ can_resume() returns True for completed agents
- ✅ Failure handling and recovery_possible flag
- ✅ Checkpoint JSON validity

#### `test_ml_agent.py` (existing, 15+ тестов)
- ✅ ModelRunner with outboxml
- ✅ FigurePlotter (creates 4 PDF figures)
- ✅ ResultExporter (model_results.json)
- ✅ MLAgent full execution
- ✅ Model metrics validation

---

## Запуск тестов

### Установка зависимостей для тестирования

```bash
pip install pytest pytest-cov
```

### Запуск всех тестов

```bash
pytest tests/
```

### Запуск с подробным выводом

```bash
pytest tests/ -v
```

### Запуск только unit tests (быстро)

```bash
pytest tests/test_*.py -v
```

### Запуск только интеграционных тестов

```bash
pytest tests/agents/ -v
```

### Запуск тестов checkpoint recovery

```bash
pytest tests/test_checkpoint_recovery.py -v
```

### Запуск одного конкретного теста

```bash
pytest tests/test_pydantic_models.py::TestRunContext::test_checkpoint_saves_to_context -v
```

### С покрытием кода (coverage)

```bash
pytest tests/ --cov=src/research_agents --cov-report=html
```

Результаты сохранятся в `htmlcov/index.html`

---

## Структура каждого теста

### Unit Test (быстрый)

```python
def test_something(tmp_path: Path) -> None:
    """Test specific behavior without I/O."""
    # Arrange
    ctx = RunContext(goal="Test", config={}, output_dir=str(tmp_path))
    
    # Act
    ctx.checkpoint(agent="test", step=1)
    
    # Assert
    assert "test" in ctx.checkpoints
```

### Integration Test (с компонентами)

```python
def test_research_agent_executes(research_agent: tuple[ResearchAgent, RunContext]) -> None:
    """Test full ResearchAgent with mocked dependencies."""
    agent, ctx = research_agent
    
    # Execute
    agent.execute()
    
    # Verify
    assert ctx.is_completed("research")
    assert ctx.artifact_path("literature_review") is not None
```

---

## Что тестируется в checkpoint механизме

### Сохранение checkpoints

```python
def test_checkpoint_saved_after_step_1():
    agent.execute()
    assert (checkpoint_dir / "queries.json").exists()
    assert ctx.checkpoints["research"].step_completed >= 1
```

### Восстановление из checkpoint

```python
def test_second_execution_skips_completed_agent():
    agent.execute()  # First run
    agent.execute()  # Second run - should skip
    # Verify no duplicate API calls made
```

### Обработка ошибок

```python
def test_checkpoint_marked_failed_on_exception():
    agent.execute()  # Fails
    assert ctx.checkpoints["research"].recovery_possible is False
```

---

## Примеры запуска перед production

### 1. Быстрая проверка (2-3 минуты)

```bash
pytest tests/test_*.py -v
# Runs all unit tests only, fast validation
```

### 2. Полная проверка (5-10 минут)

```bash
pytest tests/ -v
# Runs all tests including integrations
```

### 3. Проверка checkpoints (специфично)

```bash
pytest tests/test_checkpoint_recovery.py tests/agents/test_research_agent.py -v
# Focus on checkpoint/recovery mechanism
```

### 4. С отчётом о покрытии

```bash
pytest tests/ -v --cov=src/research_agents --cov-report=term-missing
# Shows which lines are not covered
```

---

## Мокирование компонентов

### ResearchAgent тесты используют:

- ✅ **QueryBuilder** → mocked, returns fixed 6 queries
- ✅ **Searchers** → MockSearcher с предопределёнными Papers
- ✅ **PaperAnalyzer** → mocked, всегда возвращает analysis с relevance_score > 0.6
- ✅ **Synthesizer** → mocked, возвращает фиксированные разделы

### Почему мокируем:

1. **Нет зависимости от API** (Gemini, Arxiv, etc.)
2. **Быстрое выполнение** (тест за 1-2 сек вместо 5 минут)
3. **Детерминированные результаты** (не зависит от сети)
4. **Легко тестировать ошибки** (inject RuntimeError)

---

## Интеграционные тесты на реальных данных

Если хотите полных интеграционных тестов с реальным LLM:

```python
# tests/integration/test_full_pipeline.py
@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_with_real_gemini():
    """Full end-to-end test with real Gemini API."""
    # Requires GEMINI_API_KEY in .env
    # Takes 5-10 minutes
```

Эти тесты можно запустить отдельно:

```bash
pytest tests/integration/ -v -m integration
```

---

## Примеры вывода

### Успешный запуск

```
tests/test_pydantic_models.py::TestRunContext::test_checkpoint_saves_to_context PASSED
tests/test_pydantic_models.py::TestRunContext::test_last_completed_agent_returns_last_completed PASSED
tests/agents/test_research_agent.py::TestResearchAgent::test_agent_executes_successfully PASSED

======================== 75 passed in 8.42s ========================
```

### С ошибкой

```
tests/agents/test_research_agent.py::TestResearchAgent::test_minimum_papers_requirement_enforced FAILED

AssertionError: RuntimeError not raised
```

---

## CI/CD Integration

Можно добавить в GitHub Actions:

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v
```

---

## Troubleshooting

### "ImportError: No module named 'research_agents'"

```bash
# Make sure you're in the project root and module is importable
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pytest tests/
```

### "Fixture 'tmp_path' not found"

```bash
# You have an old pytest version, upgrade:
pip install --upgrade pytest
```

### Tests hang/timeout

```bash
# Add timeout (requires pytest-timeout):
pip install pytest-timeout
pytest tests/ --timeout=30
```

---

## Перед первым запуском pipeline

✅ Запустите все тесты:

```bash
pytest tests/ -v
# Should show 75+ tests passing
```

✅ Проверьте checkpoint recovery:

```bash
pytest tests/test_checkpoint_recovery.py -v
# Should show 15+ tests passing
```

✅ Проверьте ResearchAgent:

```bash
pytest tests/agents/test_research_agent.py -v
# Should show 15+ tests passing
```

Только после этого — готовы к первому запуску:

```bash
python main.py --goal research_goal.txt --config agent_config.yaml --output ./output
```

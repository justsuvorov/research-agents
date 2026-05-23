# Checkpoint/Recovery Mechanism — Usage Guide

## Overview

The pipeline now automatically saves checkpoints after each major step in each agent. This allows the pipeline to resume from the last checkpoint without repeating work or wasting tokens.

---

## Normal Execution (Automatic Resume)

**First run:**
```bash
python main.py \
  --goal research_goal.txt \
  --config agent_config.yaml \
  --output ./output
```

**If interrupted (e.g., timeout, network error):**
```bash
# Simply run the same command again
python main.py \
  --goal research_goal.txt \
  --config agent_config.yaml \
  --output ./output
```

✅ The pipeline automatically resumes from the last completed agent, skipping agents that are already done.

---

## Checkpoint Structure

Checkpoints are saved in `output/checkpoints/{agent_name}/`:

```
output/
├── checkpoints/
│   └── research/
│       ├── queries.json           (after step 1)
│       ├── papers_raw.json        (after step 2)
│       ├── papers_dedup.json      (after step 3)
│       ├── analyses.json          (after step 4)
│       ├── sections.json          (after step 5)
│       └── (after step 6: artifacts in main output dir)
│
├── run_context.json               (tracks agent completion status)
├── literature_review.md
├── references.bib
└── papers.json
```

---

## Reset & Start Fresh

To discard checkpoints and start from scratch:

```bash
python main.py \
  --goal research_goal.txt \
  --config agent_config.yaml \
  --output ./output \
  --reset
```

This deletes:
- `output/run_context.json`
- `output/checkpoints/` directory

Then runs the pipeline from the beginning.

---

## How It Works

### For ResearchAgent:
1. **Step 1:** Generate search queries → save to `queries.json`
2. **Step 2:** Search all sources → save to `papers_raw.json`
3. **Step 3:** Deduplicate papers → save to `papers_dedup.json`
4. **Step 4:** Analyze & filter papers → save to `analyses.json`
5. **Step 5:** Synthesize review sections → save to `sections.json`
6. **Step 6:** Export to files → update `run_context.json`

After step 6 completes, the agent is marked as `completed` in `run_context.json`.

### Resume Logic:
- **BaseAgent.execute()** checks `is_completed(agent_name)`
- If agent is already completed → skip execution
- If agent is not completed → run from start
  - Intermediate checkpoints are saved for debugging/analysis
  - On next run, BaseAgent skips the agent, saving all tokens

---

## Example Scenarios

### Scenario 1: Interrupted After Step 3 (Deduplication)

```
Run 1: python main.py --goal research_goal.txt --output ./output
  → Step 1: queries ✓
  → Step 2: papers_raw ✓
  → Step 3: papers_dedup ✓
  → TIMEOUT / NETWORK ERROR / KILLED
  
Run 2: python main.py --goal research_goal.txt --output ./output
  → BaseAgent checks: is_completed("research") = False
  → ResearchAgent.run() executes again (all steps 1-6)
  → But previous checkpoints are available in output/checkpoints/research/
  
# Token usage: Full re-run with same queries, papers, analyses
```

### Scenario 2: Completed ResearchAgent, Move to DataAgent

```
Run 1: python main.py --goal research_goal.txt --output ./output
  → ResearchAgent completes all steps ✓
  → DataAgent starts...
  → TIMEOUT
  
Run 2: python main.py --goal research_goal.txt --output ./output
  → BaseAgent checks: is_completed("research") = True
  → ResearchAgent.run() SKIPPED
  → DataAgent.run() resumes ✓
  
# Token saved: 0 tokens on research (already complete)
```

### Scenario 3: Reset and Try Different Config

```
python main.py --goal research_goal.txt --output ./output --reset

# Deletes all checkpoints and run_context.json
# Pipeline starts from scratch
```

---

## Monitoring Checkpoints

Check the status of a run:

```bash
# View completion status
cat output/run_context.json | grep agent_status

# View checkpoint details
ls -la output/checkpoints/research/

# Check which step was last completed
cat output/run_context.json | grep checkpoint
```

---

## Notes

- Checkpoints are **JSON files** saved for inspection and recovery
- The `run_context.json` file is the **source of truth** for agent completion status
- Partial checkpoints (steps 1-5) are **for debugging** — the pipeline focuses on agent-level completion
- Token savings come from **agent-level skipping** (BaseAgent.execute() check)
- Use `--reset` to force a clean run from scratch

---

## Troubleshooting

### "Agent marked as completed but output files missing"
→ Delete `output/run_context.json` and re-run. The agent will re-execute and regenerate files.

### "Checkpoint directory exists but agent not resuming"
→ Check `run_context.json` — if agent status is not "completed", the agent will run from scratch.

### "I want to re-run just one agent"
→ Use `--reset` to clear everything, then configure `agent_config.yaml` to skip other agents, or manually delete the target agent's status in `run_context.json`.

---

## Future Enhancements

Planned improvements:
- [ ] Mid-step resume (e.g., resume from step 3 of 6 instead of full re-run)
- [ ] Checkpoint archival (compress old checkpoints)
- [ ] Selective agent reset (reset only specific agents)
- [ ] Token usage summary per checkpoint

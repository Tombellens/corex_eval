# corex_eval

Shared evaluation library for the CoREx LLM benchmarking project.

Provides a standardised, reproducible workflow for evaluating automated
political career data pipelines across three tasks: **collection**,
**extraction**, and **annotation**.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Tombellens/corex_eval.git
cd corex_eval
pip install -e ".[dev]"
```

For annotation semantic similarity (optional, requires GPU recommended):

```bash
pip install -e ".[dev,embeddings]"
```

### 2. Point the library at your data

The gold and silver CSV files are **not** committed to the repo.
Set the `COREX_DATA_DIR` environment variable to wherever you store them locally:

```bash
export COREX_DATA_DIR=/path/to/your/data
```

Expected structure:

```
$COREX_DATA_DIR/
├── gold/
│   └── corex_gold.csv
└── silver/
    └── corex_silver.csv    # once built
```

Add the export to your `.bashrc` or `.zshrc` so it persists across sessions.

### 3. Set your GitHub token (for result submission)

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

Generate a token at <https://github.com/settings/tokens> with `repo` scope.

---

## Quickstart

```python
from corex_eval import load_inputs, evaluate, load_training_data
```

### Collection

```python
# Get test inputs
inputs = load_inputs(task="collection")
# → DataFrame: [case_id, name_first, name_last, job_title, country_label]

# Run your model → predictions_df with columns [case_id, retrieved_urls]
# retrieved_urls: list of URL strings, or pipe-separated string

results = evaluate(predictions_df, task="collection")
```

### Extraction — atomic variable

```python
inputs = load_inputs(task="extraction")
# → DataFrame: [case_id, cv_local]

# Run your model → predictions_df with columns [case_id, birth_year]

results = evaluate(predictions_df, task="extraction", variable="birth_year")
```

### Extraction — career history (composite)

```python
inputs = load_inputs(task="extraction")

# predictions_df must have column 'career' as a list of dicts per row:
# [{"start_year": 2010, "end_year": 2015, "position": "Minister"}, ...]

results = evaluate(predictions_df, task="extraction", variable="career")
```

### Annotation

```python
inputs = load_inputs(task="annotation", variable="career_position")
# → DataFrame: [case_id, spell_index, job_description_label]

# Run your model → predictions_df with columns:
# [case_id, spell_index, predicted_code]

results = evaluate(
    predictions_df,
    task="annotation",
    variable="career_position",
    semantic_similarity=True,   # optional, requires [embeddings]
)
```

### Load training data

```python
train = load_training_data(
    task="extraction",
    features=["cv_local", "birth_year", "sex", "edu_degree"],
)
```

### Submit results to the shared leaderboard

Add `submit=True` and point to your experiment config:

```python
results = evaluate(
    predictions_df,
    task="extraction",
    variable="birth_year",
    submit=True,
    experiment_path="experiments/extraction/gpt4o/config.yaml",
)
```

---

## Available variables

| Task | Variable | Type | Metric |
|---|---|---|---|
| `extraction` | `birth_year` | int | MAE + accuracy |
| `extraction` | `birth_place` | str | Accuracy |
| `extraction` | `birth_country` | str | Accuracy |
| `extraction` | `sex` | str | Accuracy |
| `extraction` | `edu_start` | int | MAE + accuracy |
| `extraction` | `edu_end` | int | MAE + accuracy |
| `extraction` | `edu_degree` | str | Accuracy |
| `extraction` | `career` | composite | Precision / Recall / F1 |
| `annotation` | `career_position` | classification | Accuracy / F1 |
| `annotation` | `edu_degree` | classification | Accuracy / F1 |
| `annotation` | `uni_subject` | classification | Accuracy / F1 |

---

## Experiment structure

Each experiment lives in its own folder under `experiments/`:

```
experiments/
├── collection/
├── extraction/
│   └── gpt4o/
│       └── config.yaml
└── annotation/
    └── encoders/
        └── mmBERT_finetuned_career/
            └── config.yaml
```

Minimal `config.yaml`:

```yaml
task: extraction
model: gpt-4o
contributor: tom
description: Zero-shot birth year extraction from cv_local
notes: Baseline, no few-shot examples
```

---

## Running tests

```bash
pytest tests/ -v
```
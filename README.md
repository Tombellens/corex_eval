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

Optional extras:

```bash
pip install -e ".[dev,embeddings]"   # semantic similarity in annotation evaluation
pip install -e ".[dev,bert]"         # BERT fine-tuning experiments
```

### 2. Point the library at your data

The gold and silver CSV files are **not** committed to the repo.
The recommended approach is a `.env` file in the project root (already in `.gitignore`):

```bash
# .env
COREX_DATA_DIR=/path/to/your/data
GITHUB_TOKEN=ghp_your_token_here
```

Then load it at the top of your notebook or script:

```python
from dotenv import load_dotenv
load_dotenv()
```

Alternatively, export the variable in your shell:

```bash
export COREX_DATA_DIR=/path/to/your/data
```

Expected data structure:

```
$COREX_DATA_DIR/
├── gold/
│   └── corex_gold.csv
└── silver/
    ├── corex_silver.csv        # career spell rows (once built)
    └── corex_silver_edu.csv    # education rows (once built)
```

### 3. Set your GitHub token (for result submission)

Add `GITHUB_TOKEN` to your `.env` file (see above), or export it in your shell:

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

Generate a token at <https://github.com/settings/tokens> with `repo` scope.
Each contributor uses their own token.

---

## Quickstart

```python
from dotenv import load_dotenv
load_dotenv()

from corex_eval import load_inputs, evaluate, load_training_data
```

### Collection

```python
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
# results includes: accuracy, macro_f1, weighted_f1,
#                   per_class breakdown, per_country breakdown
```

#### Broad sector evaluation

Evaluate at the first-digit level (e.g. `1` = Executive triangle) instead of
fine-grained 3-digit codes:

```python
results = evaluate(
    predictions_df,
    task="annotation",
    variable="career_position",
    granularity="broad",
)
```

Use `career_position_to_sector()` to collapse labels when training on broad sectors:

```python
from corex_eval import career_position_to_sector

train_df["career_position"] = train_df["career_position"].map(career_position_to_sector)
# "105 = Minister with portfolio" → "1"
```

### Dev split and confidence intervals

Tune prompts and design choices on the dev split, never on test:

```python
inputs  = load_inputs(task="annotation", variable="career_position", split="dev")
results = evaluate(predictions_df, task="annotation", variable="career_position", split="dev")
```

The dev set (15% of the training pool) is excluded from `load_training_data()`
by default; pass `include_dev=True` for final test-set runs. Dev results
cannot be submitted.

Every evaluation includes 95% bootstrap intervals in `results["ci"]`,
resampled by person (`n_bootstrap=1000` by default, `0` disables).

### Load training data

```python
# Extraction
train = load_training_data(
    task="extraction",
    features=["cv_local", "birth_year", "sex"],
)

# Annotation
train = load_training_data(
    task="annotation",
    variable="career_position",
    features=["job_description_label", "career_position"],
)
# → DataFrame: [case_id, spell_index, job_description_label, career_position]
```

### Submit results to the shared leaderboard

Add `submit=True` and point to your experiment config:

```python
results = evaluate(
    predictions_df,
    task="annotation",
    variable="career_position",
    submit=True,
    experiment_path="experiments/annotation/bert_finetuned_career/config.yaml",
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
| `annotation` | `career_position` | classification | Accuracy / F1 / per-country |
| `annotation` | `uni_subject` | classification | Accuracy / F1 / per-country |

---

## Experiment structure

Each experiment lives in its own folder under `experiments/` and consists of
a `config.yaml` for metadata and a `notebook.ipynb` to run the experiment:

```
experiments/
├── collection/
├── extraction/
└── annotation/
    ├── bert_finetuned_career/
    │   ├── config.yaml
    │   └── notebook.ipynb
    └── bert_finetuned_career_broad/
        ├── config.yaml
        └── notebook.ipynb
```

Minimal `config.yaml`:

```yaml
task: annotation
variable: career_position
model: bert-base-multilingual-cased
contributor: tom
description: Fine-tuned mmBERT for career_position annotation
notes: 5 epochs, lr=2e-5, batch_size=32
```

---

## Running tests

```bash
python3 -m pytest tests/ -v
```

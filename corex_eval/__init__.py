"""
corex_eval
==========
Shared evaluation library for the CoREx LLM benchmarking project.

This library provides a standardised workflow for evaluating automated
political career data pipelines across three tasks:

    - collection  : URL retrieval for biographical sources
    - extraction  : Structured field extraction from biographical text
    - annotation  : Codebook-based classification of extracted entries

Quickstart
----------
    from corex_eval import load_inputs, evaluate, load_training_data

    # --- Extraction example ---

    # 1. Get test set inputs (case_id + cv_local text)
    inputs = load_inputs(task="extraction")

    # 2. Run your model on inputs → predictions_df
    #    predictions_df must have columns: [case_id, birth_year]

    # 3. Evaluate
    results = evaluate(
        predictions_df,
        task="extraction",
        variable="birth_year",
    )

    # 4. Submit to shared leaderboard (requires GITHUB_TOKEN env var)
    results = evaluate(
        predictions_df,
        task="extraction",
        variable="birth_year",
        submit=True,
        experiment_path="experiments/extraction/gpt4o/config.yaml",
    )

    # --- Load training data ---
    train = load_training_data(
        task="extraction",
        features=["cv_local", "birth_year", "sex", "edu_degree"],
    )

Setup
-----
1. Clone the repository and install:
       pip install -e ".[dev]"

2. Point the library at your local data:
       export COREX_DATA_DIR=/path/to/your/data/folder

   Expected structure:
       $COREX_DATA_DIR/
       ├── gold/
       │   └── corex_gold.csv
       └── silver/
           └── corex_silver.csv   (once built)

3. For submit=True, set your GitHub token:
       export GITHUB_TOKEN=ghp_your_token_here

4. For semantic similarity in annotation evaluation:
       pip install corex_eval[embeddings]

Tasks and variables
-------------------
    collection
        No variable needed.
        predictions_df columns: [case_id, retrieved_urls]

    extraction
        Atomic variables (single value per person):
            birth_year, birth_place, birth_country,
            sex, edu_start, edu_end, edu_degree

        Composite variables (full record list per person):
            career   → predictions_df column 'career' must be a list of
                        dicts: [{"start_year": int, "end_year": int,
                                 "position": str}, ...]

    annotation
        Variables: career_position, edu_degree, uni_subject
        predictions_df columns: [case_id, spell_index, predicted_code]
        (spell_index not required for edu_degree)
"""

from corex_eval.inputs import load_inputs, load_training_data
from corex_eval.evaluate import evaluate

__all__ = [
    "load_inputs",
    "load_training_data",
    "evaluate",
]

__version__ = "0.1.0"
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
    from dotenv import load_dotenv
    load_dotenv()   # loads COREX_DATA_DIR and GITHUB_TOKEN from .env

    from corex_eval import load_inputs, evaluate, load_training_data

    # --- Annotation example ---

    # 1. Get test set inputs
    inputs = load_inputs(task="annotation", variable="career_position")
    # → DataFrame: [case_id, spell_index, job_description_label]

    # 2. Run your model on inputs → predictions_df
    #    predictions_df must have columns: [case_id, spell_index, predicted_code]

    # 3. Evaluate (returns accuracy, macro_f1, per_class, per_country, ...)
    results = evaluate(
        predictions_df,
        task="annotation",
        variable="career_position",
    )

    # 4. Evaluate at broad sector level (first digit only)
    results = evaluate(
        predictions_df,
        task="annotation",
        variable="career_position",
        granularity="broad",
    )

    # 5. Submit to shared leaderboard (requires GITHUB_TOKEN in .env)
    results = evaluate(
        predictions_df,
        task="annotation",
        variable="career_position",
        submit=True,
        experiment_path="experiments/annotation/bert_finetuned_career/config.yaml",
    )

    # --- Load training data ---
    train = load_training_data(
        task="annotation",
        variable="career_position",
        features=["job_description_label", "career_position"],
    )
    # → DataFrame: [case_id, spell_index, job_description_label, career_position]

    # --- Collapse fine-grained codes to broad sectors for training ---
    from corex_eval import career_position_to_sector
    train["career_position"] = train["career_position"].map(career_position_to_sector)
    # "105 = Minister with portfolio" → "1"

Setup
-----
1. Clone the repository and install:
       pip install -e ".[dev]"

   Optional extras:
       pip install -e ".[dev,embeddings]"   # semantic similarity
       pip install -e ".[dev,bert]"         # BERT fine-tuning experiments

2. Create a .env file in the project root (gitignored):
       COREX_DATA_DIR=/path/to/your/data
       GITHUB_TOKEN=ghp_your_token_here

   Expected data structure:
       $COREX_DATA_DIR/
       ├── gold/
       │   └── corex_gold.csv
       └── silver/
           ├── corex_silver.csv        (career rows, once built)
           └── corex_silver_edu.csv    (education rows, once built)

3. In notebooks, load .env before importing corex_eval:
       from dotenv import load_dotenv
       load_dotenv()

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
        Variables: career_position, uni_subject
        predictions_df columns: [case_id, spell_index, predicted_code]

        career_position supports granularity="broad" to evaluate at the
        broad sector level (first digit) instead of fine-grained 3-digit codes.
        Results always include per_class and per_country breakdowns.
"""

from corex_eval.inputs import load_inputs, load_training_data
from corex_eval.evaluate import evaluate
from corex_eval.config import career_position_to_sector, CAREER_POSITION_SECTORS, CAREER_POSITION_SECTOR_HINTS
from corex_eval.config import uni_subject_to_prefix

__all__ = [
    "load_inputs",
    "load_training_data",
    "evaluate",
    "career_position_to_sector",
    "CAREER_POSITION_SECTORS",
    "CAREER_POSITION_SECTOR_HINTS",
    "uni_subject_to_prefix",
]

__version__ = "0.1.0"
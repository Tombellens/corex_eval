"""
build_silver.py
===============
Build the CoREx silver standard dataset using GPT-4o.

Two modes are supported via the --mode flag:

  career (default)
    Extracts workplace and job title information from CV text.
    For each politician, we provide:
      - Their CV text (cv_english with cv_local fallback)
      - The spell anchors from the gold standard (spell_index, start_year, end_year)
    GPT-4o returns for each spell:
      - job_description_label: the raw job title / position description
      - workplace_label: the organisation / institution name
    Output: data/silver/corex_silver.csv

  edu
    Extracts education descriptions (degree and subject labels) from CV text.
    For each politician, we provide:
      - Their CV text (cv_english with cv_local fallback)
      - The education anchor: coded edu_degree + coded uni_subject fields
    GPT-4o returns:
      - degree_label: raw degree name from the CV
      - Per subject: subject_label combining degree type and field of study
    Output: data/silver/corex_silver_edu.csv

Output
------
  data/silver/corex_silver.csv       — career silver (successful extractions)
  data/silver/failures.csv           — career failures, for a second pass
  data/silver/corex_silver_edu.csv   — education silver (successful extractions)
  data/silver/edu_failures.csv       — education failures, for a second pass

Usage
-----
  python build_silver.py [--mode career|edu]

  Optional flags:
    --limit N         only process first N cases (for testing)
    --case-ids X Y Z  only process specific case_ids
    --second-pass     read failures.csv (or edu_failures.csv) and retry

Requirements
------------
  pip install openai pandas tqdm
  export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — adjust if your COREX_DATA_DIR differs
# ---------------------------------------------------------------------------

DATA_DIR      = Path(os.environ.get("COREX_DATA_DIR", "data"))
GOLD_PATH     = DATA_DIR / "gold" / "corex_gold.csv"
SILVER_DIR    = DATA_DIR / "silver"
SILVER_OUT    = SILVER_DIR / "corex_silver.csv"
FAILURES_OUT  = SILVER_DIR / "failures.csv"

SILVER_EDU_OUT   = SILVER_DIR / "corex_silver_edu.csv"
EDU_FAILURES_OUT = SILVER_DIR / "edu_failures.csv"

# ---------------------------------------------------------------------------
# GPT settings
# ---------------------------------------------------------------------------

MODEL         = "gpt-4o"
MAX_TOKENS    = 1500
TEMPERATURE   = 0.0          # deterministic
RETRY_WAIT    = 5            # seconds to wait between retries
MAX_RETRIES   = 3

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise data extraction assistant. You will be given:
1. A politician's CV text
2. A list of career spell anchors, each with a spell_index and year range

Your task is to find the job title and workplace for each spell anchor by matching
the year range to the correct position in the CV.

RULES:
- Return ONLY valid JSON, no explanation, no markdown, no code fences.
- For each spell, return job_description_label and workplace_label as plain strings.
- job_description_label: the job title or role (e.g. "Secretary", "Member of Parliament")
- workplace_label: the organisation or institution (e.g. "Municipality of Ulcinj", "Parliament of Montenegro")
- If you cannot confidently match a spell to the CV text, set both fields to null and set "extraction_failed" to true for that spell.
- Never invent information. If unsure, return null.
- Preserve the exact spell_index values from the input — do not renumber them.

Output format:
{
  "spells": [
    {
      "spell_index": 1,
      "job_description_label": "...",
      "workplace_label": "...",
      "extraction_failed": false
    },
    ...
  ]
}"""


def build_user_prompt(cv_text: str, spells: list[dict]) -> str:
    spell_lines = "\n".join(
        f"  - spell_index {s['spell_index']}: "
        f"{s['start_year'] if s['start_year'] else '?'} – {s['end_year'] if s['end_year'] else '?'}"
        + (f"  [category: {s['category']}]" if s.get('category') else "")
        for s in spells
    )
    return f"""CV TEXT:
{cv_text}

CAREER SPELL ANCHORS TO EXTRACT:
{spell_lines}

The category hint tells you what kind of role to look for. Use it to
disambiguate when multiple positions overlap in time or the CV is ambiguous.

Return the JSON object as specified."""


# ---------------------------------------------------------------------------
# GPT call
# ---------------------------------------------------------------------------

def _call_gpt_raw(system_prompt: str, user_prompt: str, client) -> dict | None:
    """
    Low-level GPT-4o call. Returns parsed JSON dict or None if all retries fail.
    Shared by both career and education pipelines.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if model ignores instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"    [attempt {attempt}] JSON parse error: {e}")
        except Exception as e:
            print(f"    [attempt {attempt}] API error: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT)

    return None  # all retries exhausted


def call_gpt(cv_text: str, spells: list[dict], client) -> dict | None:
    """Call GPT-4o for a career case. Returns parsed JSON dict or None on failure."""
    return _call_gpt_raw(SYSTEM_PROMPT, build_user_prompt(cv_text, spells), client)


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------

def process_result(case_id: str, result: dict, spell_index_set: set[int]) -> tuple[list[dict], list[dict]]:
    """
    Parse GPT result into silver rows and per-spell failures.

    Returns (silver_rows, failed_spells) where failed_spells is a list of
    {case_id, spell_index, reason} dicts.
    """
    silver_rows  = []
    failed_spells = []

    returned_indices = set()

    for spell in result.get("spells", []):
        idx = spell.get("spell_index")
        if idx not in spell_index_set:
            # GPT hallucinated a spell_index we didn't ask for — ignore
            continue

        returned_indices.add(idx)
        failed = spell.get("extraction_failed", False)
        jd     = spell.get("job_description_label")
        wp     = spell.get("workplace_label")

        if failed or (jd is None and wp is None):
            failed_spells.append({
                "case_id":     case_id,
                "spell_index": idx,
                "reason":      "extraction_failed_by_model",
            })
        else:
            silver_rows.append({
                "case_id":               case_id,
                "spell_index":           idx,
                "job_description_label": jd or "",
                "workplace_label":       wp or "",
            })

    # Spells GPT didn't return at all
    for idx in spell_index_set - returned_indices:
        failed_spells.append({
            "case_id":     case_id,
            "spell_index": idx,
            "reason":      "spell_not_returned_by_model",
        })

    return silver_rows, failed_spells


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_cases(case_ids_filter: list[str] | None = None) -> pd.DataFrame:
    """
    Load gold and build a per-case dataframe with cv_english and spell anchors.
    Only includes spells that have at least one year (start or end).
    """
    # We load raw CSV directly here to avoid the library's lru_cache and
    # to access cv_english without it being filtered out by load_gold()
    # Auto-detect separator (comma or semicolon)
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    df = pd.read_csv(GOLD_PATH, sep=sep, dtype=str, keep_default_na=False)

    # Drop incomplete records (missing_type set)
    if "missing_type" in df.columns:
        df = df[df["missing_type"].str.strip() == ""]

    # Drop rows with no case_id
    df = df[df["case_id"].str.strip() != ""]

    # Drop duplicate case_ids (keep first)
    df = df.drop_duplicates(subset="case_id", keep="first")

    if case_ids_filter:
        df = df[df["case_id"].isin(case_ids_filter)]

    return df


def extract_spells_for_case(row: pd.Series) -> list[dict]:
    """
    Extract spell anchors from a gold row, skipping spells with no years.
    Returns list of {spell_index, start_year, end_year}.
    """
    spells = []
    for n in range(1, 21):
        pos_col   = f"career_position_{n}"
        start_col = f"career_start_year_{n}"
        end_col   = f"career_end_year_{n}"

        pos = str(row.get(pos_col, "")).strip()
        if not pos or pos in ("", "nan"):
            continue

        start = row.get(start_col, "")
        end   = row.get(end_col,   "")

        try:
            start_int = int(float(start)) if start and start not in ("", "nan") else None
        except (ValueError, TypeError):
            start_int = None

        try:
            end_int = int(float(end)) if end and end not in ("", "nan") else None
        except (ValueError, TypeError):
            end_int = None

        # Skip spells with no temporal info — unevaluable
        if start_int is None and end_int is None:
            continue

        spells.append({
            "spell_index": n,
            "start_year":  start_int,
            "end_year":    end_int,
            "category":    pos,   # gold career_position code, e.g. "401 = politics, parliament"
        })

    return spells


# ---------------------------------------------------------------------------
# Education prompt
# ---------------------------------------------------------------------------

EDU_SYSTEM_PROMPT = """You are a precise data extraction assistant. You will be given:
1. A politician's CV text
2. An education anchor with a coded degree level and coded subject fields

Your task is to extract:
  - degree_label: the raw degree name as it appears in the CV text
  - For each coded subject: a combined description of the degree type and field of
    study as it appears in the CV text (e.g. "Master of Political Sciences",
    "Bachelor in Applied Economics")

RULES:
- Return ONLY valid JSON, no explanation, no markdown, no code fences.
- degree_label: the degree title from the CV (e.g. "Master in Political Sciences",
  "PhD in Law"). Return null if not clearly present in the CV.
- subject_label: combine the degree type and subject field as they appear together
  in the CV. Return null and set extraction_failed to true if not found.
- Never invent information. Return null if unsure.
- Preserve the exact subject_index values from the input.

Output format:
{
  "degree_label": "...",
  "subjects": [
    {
      "subject_index": 1,
      "subject_label": "...",
      "extraction_failed": false
    }
  ]
}"""


def build_edu_user_prompt(cv_text: str, edu_anchor: dict) -> str:
    header_lines = [f"Degree level: {edu_anchor['edu_degree']}"]
    start = edu_anchor.get("edu_start")
    end   = edu_anchor.get("edu_end")
    if start or end:
        header_lines.append(f"Period: {start if start else '?'} – {end if end else '?'}")

    subject_lines = "\n".join(
        f"  - subject_index {s['subject_index']}: {s['code']}"
        + (f"  [university: {s['uni_name']}]" if s.get("uni_name") else "")
        for s in edu_anchor["subjects"]
    )

    return (
        f"CV TEXT:\n{cv_text}\n\n"
        f"EDUCATION ANCHOR:\n" + "\n".join(header_lines) + "\n\n"
        f"SUBJECT ANCHORS TO EXTRACT:\n{subject_lines}\n\n"
        f"The category hint tells you what field of study to look for.\n"
        f"Return the JSON object as specified."
    )


def extract_edu_for_case(row: pd.Series) -> dict | None:
    """
    Extract the education anchor from a gold row.

    Returns a dict with edu_degree, edu_start, edu_end, and a subjects list,
    or None if there are no subjects to extract.
    """
    edu_degree = str(row.get("edu_degree", "")).strip()
    edu_degree = edu_degree if edu_degree and edu_degree != "nan" else None

    try:
        raw = row.get("edu_start", "")
        start_int = int(float(raw)) if raw and str(raw) not in ("", "nan") else None
    except (ValueError, TypeError):
        start_int = None

    try:
        raw = row.get("edu_end", "")
        end_int = int(float(raw)) if raw and str(raw) not in ("", "nan") else None
    except (ValueError, TypeError):
        end_int = None

    subjects = []
    for n in range(1, 6):
        subj = str(row.get(f"uni_subject_{n}", "")).strip()
        if not subj or subj == "nan":
            continue
        uni_name = str(row.get(f"uni_name_{n}", "")).strip()
        subjects.append({
            "subject_index": n,
            "code":          subj,
            "uni_name":      uni_name if uni_name and uni_name != "nan" else None,
        })

    if not subjects:
        return None  # nothing to extract without at least one coded subject

    return {
        "edu_degree": edu_degree or "unknown",
        "edu_start":  start_int,
        "edu_end":    end_int,
        "subjects":   subjects,
    }


def process_edu_result(
    case_id: str,
    result: dict,
    index_to_code: dict[int, str],
) -> tuple[list[dict], list[dict]]:
    """
    Parse GPT education result into silver rows and failure records.

    Output silver rows:
      degree row  → {case_id, degree_label, uni_subject="", subject_label=""}
      subject rows → {case_id, degree_label="", uni_subject=<gold code>, subject_label}

    The original gold uni_subject code is stored directly in each subject row
    (keyed by subject_index via index_to_code), so traceability from extracted
    label to gold annotation is preserved without relying on positional indices.

    Failure records still use spell_index (0 = degree, 1–5 = subject position)
    for internal retry tracking only.

    Returns (silver_rows, failed_entries).
    """
    silver_rows    = []
    failed_entries = []

    expected_indices = set(index_to_code.keys())

    degree_label = (result.get("degree_label") or "").strip()

    if degree_label:
        silver_rows.append({
            "case_id":       case_id,
            "degree_label":  degree_label,
            "uni_subject":   "",
            "subject_label": "",
        })
    else:
        failed_entries.append({
            "case_id":     case_id,
            "spell_index": 0,
            "reason":      "degree_label_not_found",
        })

    returned_indices: set[int] = set()

    for subj in result.get("subjects", []):
        idx = subj.get("subject_index")
        if idx not in expected_indices:
            continue  # GPT hallucinated an index we didn't ask for

        returned_indices.add(idx)
        failed = subj.get("extraction_failed", False)
        sl     = subj.get("subject_label")

        if failed or sl is None:
            failed_entries.append({
                "case_id":     case_id,
                "spell_index": idx,
                "reason":      "extraction_failed_by_model",
            })
        else:
            silver_rows.append({
                "case_id":       case_id,
                "degree_label":  "",
                "uni_subject":   index_to_code[idx],
                "subject_label": sl or "",
            })

    for idx in expected_indices - returned_indices:
        failed_entries.append({
            "case_id":     case_id,
            "spell_index": idx,
            "reason":      "subject_not_returned_by_model",
        })

    return silver_rows, failed_entries


# ---------------------------------------------------------------------------
# Education pipeline
# ---------------------------------------------------------------------------

def run_edu_pipeline(
    limit: int | None = None,
    case_ids_filter: list[str] | None = None,
    second_pass: bool = False,
):
    # --- Setup ---
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load cases ---
    if second_pass:
        if not EDU_FAILURES_OUT.exists():
            print(f"No edu failures file found at {EDU_FAILURES_OUT}. Nothing to retry.")
            return
        failures_df = pd.read_csv(EDU_FAILURES_OUT, dtype=str)
        case_ids_filter = failures_df["case_id"].unique().tolist()
        print(f"Second pass: retrying {len(case_ids_filter)} failed case_ids.")

    gold_df = load_cases(case_ids_filter)

    if limit:
        gold_df = gold_df.head(limit)

    print(f"Processing {len(gold_df)} cases (education) with model {MODEL}.")

    cv_col = "cv_english" if "cv_english" in gold_df.columns else "cv_local"
    if cv_col == "cv_local":
        print("Warning: cv_english not found, falling back to cv_local.")

    # --- Load existing edu silver to allow resuming ---
    if SILVER_EDU_OUT.exists():
        existing_silver = pd.read_csv(SILVER_EDU_OUT, dtype=str)
        already_done    = set(existing_silver["case_id"].tolist())
        print(f"Resuming: {len(already_done)} case_ids already in edu silver file.")
    else:
        existing_silver = pd.DataFrame()
        already_done    = set()

    # --- Process ---
    all_silver_rows    = []
    all_failed_entries = []
    skipped = 0

    for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Extracting edu"):
        case_id = str(row["case_id"]).strip()

        if case_id in already_done:
            skipped += 1
            continue

        cv_text = str(row.get(cv_col, "")).strip()
        if not cv_text or cv_text == "nan":
            all_failed_entries.append({
                "case_id":     case_id,
                "spell_index": "ALL",
                "reason":      "no_cv_text",
            })
            continue

        edu_anchor = extract_edu_for_case(row)
        if edu_anchor is None:
            skipped += 1  # no coded subjects — nothing to extract
            continue

        index_to_code = {s["subject_index"]: s["code"] for s in edu_anchor["subjects"]}
        user_prompt   = build_edu_user_prompt(cv_text, edu_anchor)
        result        = _call_gpt_raw(EDU_SYSTEM_PROMPT, user_prompt, client)

        if result is None:
            for idx in index_to_code:
                all_failed_entries.append({
                    "case_id":     case_id,
                    "spell_index": idx,
                    "reason":      "api_call_failed_all_retries",
                })
            all_failed_entries.append({
                "case_id":     case_id,
                "spell_index": 0,
                "reason":      "api_call_failed_all_retries",
            })
            continue

        silver_rows, failed_entries = process_edu_result(case_id, result, index_to_code)
        all_silver_rows.extend(silver_rows)
        all_failed_entries.extend(failed_entries)

    # --- Write outputs ---
    if skipped:
        print(f"Skipped {skipped} cases (already done or no education data).")

    if all_silver_rows:
        new_silver = pd.DataFrame(all_silver_rows)
        combined   = pd.concat([existing_silver, new_silver], ignore_index=True)
        combined.to_csv(SILVER_EDU_OUT, index=False)
        print(f"Wrote {len(all_silver_rows)} new edu silver rows → {SILVER_EDU_OUT}")
        print(f"Total edu silver rows: {len(combined)}")
    else:
        print("No new edu silver rows to write.")

    if all_failed_entries:
        failures_df = pd.DataFrame(all_failed_entries)
        failures_df.to_csv(EDU_FAILURES_OUT, index=False)
        print(f"Wrote {len(all_failed_entries)} failures → {EDU_FAILURES_OUT}")
        print("Run with --mode edu --second-pass to retry.")
    else:
        print("No failures.")
        if EDU_FAILURES_OUT.exists():
            EDU_FAILURES_OUT.unlink()
            print("Cleared previous edu failures file.")


# ---------------------------------------------------------------------------
# Career pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    limit: int | None = None,
    case_ids_filter: list[str] | None = None,
    second_pass: bool = False,
):
    # --- Setup ---
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load cases ---
    if second_pass:
        if not FAILURES_OUT.exists():
            print(f"No failures file found at {FAILURES_OUT}. Nothing to retry.")
            return
        failures_df = pd.read_csv(FAILURES_OUT, dtype=str)
        case_ids_filter = failures_df["case_id"].unique().tolist()
        print(f"Second pass: retrying {len(case_ids_filter)} failed case_ids.")

    gold_df = load_cases(case_ids_filter)

    if limit:
        gold_df = gold_df.head(limit)

    print(f"Processing {len(gold_df)} cases with model {MODEL}.")

    # --- Load existing silver to allow resuming ---
    if SILVER_OUT.exists():
        existing_silver = pd.read_csv(SILVER_OUT, dtype=str)
        already_done = set(existing_silver["case_id"].tolist())
        print(f"Resuming: {len(already_done)} case_ids already in silver file.")
    else:
        existing_silver = pd.DataFrame()
        already_done    = set()

    # --- Process ---
    all_silver_rows  = []
    all_failed_spells = []
    skipped = 0

    cv_col = "cv_english" if "cv_english" in gold_df.columns else "cv_local"
    if cv_col == "cv_local":
        print(f"Warning: cv_english not found, falling back to cv_local.")

    for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Extracting"):
        case_id = str(row["case_id"]).strip()

        # Skip if already processed
        if case_id in already_done:
            skipped += 1
            continue

        cv_text = str(row.get(cv_col, "")).strip()
        if not cv_text or cv_text in ("", "nan"):
            all_failed_spells.append({
                "case_id":     case_id,
                "spell_index": "ALL",
                "reason":      "no_cv_text",
            })
            continue

        spells = extract_spells_for_case(row)
        if not spells:
            skipped += 1  # no evaluable spells — nothing to extract
            continue

        spell_index_set = {s["spell_index"] for s in spells}

        # Call GPT
        result = call_gpt(cv_text, spells, client)

        if result is None:
            # All retries failed
            for s in spells:
                all_failed_spells.append({
                    "case_id":     case_id,
                    "spell_index": s["spell_index"],
                    "reason":      "api_call_failed_all_retries",
                })
            continue

        silver_rows, failed_spells = process_result(case_id, result, spell_index_set)
        all_silver_rows.extend(silver_rows)
        all_failed_spells.extend(failed_spells)

    # --- Write outputs ---
    if skipped:
        print(f"Skipped {skipped} cases (already done or no evaluable spells).")

    # Append new silver rows to existing file
    if all_silver_rows:
        new_silver = pd.DataFrame(all_silver_rows)
        combined   = pd.concat([existing_silver, new_silver], ignore_index=True)
        combined.to_csv(SILVER_OUT, index=False)
        print(f"Wrote {len(all_silver_rows)} new silver rows → {SILVER_OUT}")
        print(f"Total silver rows: {len(combined)}")
    else:
        print("No new silver rows to write.")

    # Write failures (overwrite — second pass replaces previous failures)
    if all_failed_spells:
        failures_df = pd.DataFrame(all_failed_spells)
        failures_df.to_csv(FAILURES_OUT, index=False)
        print(f"Wrote {len(all_failed_spells)} failures → {FAILURES_OUT}")
        print("Run with --second-pass to retry.")
    else:
        print("No failures.")
        if FAILURES_OUT.exists():
            FAILURES_OUT.unlink()
            print("Cleared previous failures file.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CoREx silver standard via GPT-4o")
    parser.add_argument(
        "--mode", choices=["career", "edu"], default="career",
        help="Which silver standard to build: 'career' (default) or 'edu'",
    )
    parser.add_argument("--limit",       type=int,  default=None, help="Only process first N cases")
    parser.add_argument("--case-ids",    nargs="+", default=None, help="Only process these case_ids")
    parser.add_argument("--second-pass", action="store_true",
                        help="Retry failed cases (reads failures.csv or edu_failures.csv)")
    args = parser.parse_args()

    if args.mode == "edu":
        run_edu_pipeline(
            limit           = args.limit,
            case_ids_filter = args.case_ids,
            second_pass     = args.second_pass,
        )
    else:
        run_pipeline(
            limit           = args.limit,
            case_ids_filter = args.case_ids,
            second_pass     = args.second_pass,
        )
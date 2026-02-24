"""
submit.py
=========
Submit evaluation results to the shared GitHub results register.

When a contributor calls evaluate(..., submit=True), this module:
  1. Reads their experiment config.yaml for metadata
  2. Flattens the results dict into a single CSV row
  3. Fetches the current register.csv from GitHub
  4. Appends the new row
  5. Pushes the updated file back via the GitHub Contents API

No local git commands are used — everything goes through the GitHub
REST API with a personal access token.

Setup (one-time per contributor)
---------------------------------
1. Create a GitHub personal access token with repo write access:
   https://github.com/settings/tokens
2. Set it as an environment variable:
   export GITHUB_TOKEN=ghp_your_token_here
   (add this to your .bashrc or .zshrc)
3. Update GITHUB_REPO_OWNER and GITHUB_REPO_NAME in config.py
   to match your actual repository.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from corex_eval.config import (
    GITHUB_REPO_NAME,
    GITHUB_REPO_OWNER,
    GITHUB_RESULTS_FILE_PATH,
    GITHUB_TOKEN_ENV_VAR,
    RESULTS_PATH,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit_results(
    results: dict[str, Any],
    experiment_path: str | Path,
) -> None:
    """
    Append evaluation results to the shared GitHub register.

    Reads contributor and model metadata from the experiment config.yaml,
    combines it with the results dict, and appends one row to
    results/register.csv on GitHub.

    Parameters
    ----------
    results         : The dict returned by evaluate(). Must contain at
                      least "task" and "variable".
    experiment_path : Path to the experiment config.yaml that produced
                      these results. Used to extract metadata.

    Raises
    ------
    EnvironmentError  : If GITHUB_TOKEN is not set.
    FileNotFoundError : If experiment_path does not exist.
    RuntimeError      : If the GitHub API call fails.
    """
    token = os.environ.get(GITHUB_TOKEN_ENV_VAR)
    if not token:
        raise EnvironmentError(
            f"GitHub token not found. Please set the {GITHUB_TOKEN_ENV_VAR} "
            f"environment variable:\n"
            f"  export {GITHUB_TOKEN_ENV_VAR}=ghp_your_token_here\n"
            f"Generate a token at: https://github.com/settings/tokens\n"
            f"(needs 'repo' scope for write access)"
        )

    config    = _load_experiment_config(experiment_path)
    row       = _build_row(results, config, experiment_path)
    row_line  = _row_to_csv_line(row)

    _push_to_github(row_line, token)

    print(
        f"[corex_eval] Results submitted to GitHub register.\n"
        f"  Repo      : {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}\n"
        f"  File      : {GITHUB_RESULTS_FILE_PATH}\n"
        f"  Task      : {row['task']} / {row['variable']}\n"
        f"  Contributor: {row['contributor']}\n"
        f"  Timestamp : {row['timestamp']}"
    )


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _build_row(
    results: dict[str, Any],
    config: dict[str, Any],
    experiment_path: str | Path,
) -> dict[str, Any]:
    """
    Flatten results + config metadata into a single register row.

    The register uses a fixed set of columns so all rows are comparable.
    Metric values are stored as JSON strings in the "metrics_json" column
    so the CSV stays flat while preserving full detail.
    """
    task     = results.get("task", "")
    variable = results.get("variable", "")

    # Top-level summary metrics — pulled out for easy filtering in the CSV
    summary = _extract_summary_metrics(results, task)

    return {
        # --- Identity ---
        "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
        "contributor":     config.get("contributor", "unknown"),
        "experiment_path": str(experiment_path),

        # --- Task ---
        "task":     task,
        "variable": variable,

        # --- Model ---
        "model":       config.get("model", "unknown"),
        "model_notes": config.get("notes", ""),

        # --- Summary metrics (flat, for quick comparison) ---
        **summary,

        # --- Full results (nested, for complete reproducibility) ---
        "metrics_json": json.dumps(
            {k: v for k, v in results.items()
             if k not in ("task", "variable")},
            default=str,
        ),
    }


def _extract_summary_metrics(results: dict, task: str) -> dict[str, Any]:
    """
    Pull the most important metrics out as flat columns.

    These become directly readable columns in the CSV register,
    so contributors can sort/filter without parsing metrics_json.
    """
    summary: dict[str, Any] = {
        "n_evaluated": results.get("n_evaluated"),
        "n_skipped":   results.get("n_skipped"),
    }

    if task == "collection":
        macro = results.get("macro", {})
        summary["precision"] = macro.get("precision")
        summary["recall"]    = macro.get("recall")
        summary["f1"]        = macro.get("f1")

    elif task == "extraction":
        if "mae" in results:
            # Year / integer variable
            summary["mae"]      = results.get("mae")
            summary["accuracy"] = results.get("accuracy")
        elif "macro" in results:
            # Composite variable (career)
            macro = results.get("macro", {})
            summary["precision"] = macro.get("precision")
            summary["recall"]    = macro.get("recall")
            summary["f1"]        = macro.get("f1")
        else:
            # String atomic variable
            summary["accuracy"] = results.get("accuracy")

    elif task == "annotation":
        summary["accuracy"]            = results.get("accuracy")
        summary["macro_f1"]            = results.get("macro_f1")
        summary["weighted_f1"]         = results.get("weighted_f1")
        summary["semantic_similarity"] = results.get("semantic_similarity")

    return summary


# ---------------------------------------------------------------------------
# CSV formatting
# ---------------------------------------------------------------------------

# Fixed column order for the register — every row uses the same columns
# so the CSV is always consistently shaped.
_REGISTER_COLUMNS = [
    "timestamp",
    "contributor",
    "experiment_path",
    "task",
    "variable",
    "model",
    "model_notes",
    "n_evaluated",
    "n_skipped",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "mae",
    "macro_f1",
    "weighted_f1",
    "semantic_similarity",
    "metrics_json",
]


def _row_to_csv_line(row: dict[str, Any]) -> str:
    """Serialise a row dict to a single CSV line string (no newline)."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=_REGISTER_COLUMNS,
        extrasaction="ignore",   # ignore any keys not in _REGISTER_COLUMNS
        lineterminator="",
    )
    # Fill missing columns with empty string
    full_row = {col: row.get(col, "") for col in _REGISTER_COLUMNS}
    writer.writerow(full_row)
    return buf.getvalue()


def _header_line() -> str:
    """Return the CSV header line."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=_REGISTER_COLUMNS,
        lineterminator="",
    )
    writer.writeheader()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------

def _push_to_github(new_row_line: str, token: str) -> None:
    """
    Fetch the current register.csv from GitHub, append a row, and push back.

    Uses the GitHub Contents API:
    GET  /repos/{owner}/{repo}/contents/{path}  → fetch current content + SHA
    PUT  /repos/{owner}/{repo}/contents/{path}  → update with new content

    The SHA from the GET is required for the PUT to prevent conflicts.
    """
    import urllib.request

    api_base = (
        f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}"
        f"/contents/{GITHUB_RESULTS_FILE_PATH}"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
        "Content-Type":  "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # --- Fetch current file ---
    try:
        req      = urllib.request.Request(api_base, headers=headers)
        response = urllib.request.urlopen(req, timeout=15)
        data     = json.loads(response.read().decode())
        current_content = base64.b64decode(data["content"]).decode("utf-8")
        sha             = data["sha"]
        file_exists     = True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # File doesn't exist yet — create it with header + first row
            current_content = ""
            sha             = None
            file_exists     = False
        else:
            raise RuntimeError(
                f"GitHub API error when fetching register: "
                f"HTTP {e.code} — {e.reason}"
            ) from e

    # --- Build new content ---
    if not file_exists or not current_content.strip():
        # New file: write header + row
        new_content = _header_line() + "\n" + new_row_line + "\n"
    else:
        # Existing file: append row (ensure file ends with newline first)
        existing = current_content.rstrip("\n")
        new_content = existing + "\n" + new_row_line + "\n"

    # --- Push updated content ---
    encoded   = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")
    commit_msg = (
        f"Add result: {new_row_line.split(',')[3]} / "   # task
        f"{new_row_line.split(',')[4]} "                  # variable
        f"[{new_row_line.split(',')[1]}]"                 # contributor
    )

    payload = {
        "message": commit_msg,
        "content": encoded,
    }
    if sha:
        payload["sha"] = sha

    req = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="PUT",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(
            f"GitHub API error when pushing register update: "
            f"HTTP {e.code} — {e.reason}\n{body}"
        ) from e


# ---------------------------------------------------------------------------
# Experiment config loader
# ---------------------------------------------------------------------------

def _load_experiment_config(path: str | Path) -> dict[str, Any]:
    """
    Load and return the experiment config.yaml as a plain dict.

    Expected fields (all optional but recommended):
        contributor : your name / GitHub handle
        model       : model identifier string
        notes       : free-text description of the experiment
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {resolved}\n"
            f"Make sure you pass the path to your config.yaml, e.g.:\n"
            f"  experiments/extraction/gpt4o/config.yaml"
        )
    with resolved.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}
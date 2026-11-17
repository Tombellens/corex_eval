# Workplan — post-PolMeth iteration

Sources: `TODO.tex` from the paper, plus the review notes agreed on 2026-11-17.
Ordering rule: anything that changes how results are measured or stored comes
before any new model run, so the next round of numbers is produced once.

Scope decision: the **collection task is cut from the experiments**. It stays in
the paper as a defined stage with specified metrics, left to future work.

---

## Phase 0 — Evaluation foundations (code only, no API cost) — DONE 2026-11-17

Notes from implementation:
- 0.2: the soft position-text match was dropped — gold spells only carry a
  codebook code, no job title to compare against. Strict = optimal matching
  + missing year only matches missing year.
- 0.4: also counts `n_format_mismatch` for fine career codes (right numeric
  code, different string, scored wrong). Check this on the old Gemini run.
- 0.5: the extraction prompt asks for a free-text country name but its
  example shows `17 = Belgium`; both formats are now normalised. Review the
  alias table in `config.py` against the real gold country labels.
- Not yet verified on the real data (COREX_DATA_DIR not available here).

| # | Task | Where | Done when |
|---|---|---|---|
| 0.1 | **Dev split.** Carve a dev set out of the current train split, stratified by country. Test IDs must not change. Add a `split=` argument to `load_inputs` / `evaluate`. | `corex_eval/split.py`, `inputs.py`, `evaluate.py` | A test proves the test-set IDs are identical to today's; dev and train are disjoint. |
| 0.2 | **Stricter career-extraction metric**, reported next to the current one: optimal (Hungarian) matching instead of first-fit, a soft position-text match, and missing years no longer matching everything. | `corex_eval/metrics/extraction.py` | Both variants returned by `evaluate`; old numbers reproducible. |
| 0.3 | **Bootstrap confidence intervals**, resampled by person (`case_id`), for all headline metrics. | `corex_eval/metrics/` | Every headline metric has a 95% interval. |
| 0.4 | **Invalid-output rate** for annotation: unparseable or out-of-codebook predictions counted as wrong and reported separately. | `corex_eval/evaluate.py`, `metrics/annotation.py` | `invalid_rate` in results. |
| 0.5 | **Birth country**: normalise to country code and score by exact match (character similarity kept as secondary). | `corex_eval/metrics/extraction.py`, `config.py` | "Austria" vs "Australia" scores 0. |

## Phase 1 — Infrastructure

| # | Task | Where | Done when |
|---|---|---|---|
| 1.1 | **Store full predictions** per run (case, spell/subject key, country, gold, predicted, raw model output). | `corex_eval/evaluate.py`, `submit.py` | One predictions file per run. |
| 1.2 | **Results out of git.** Replace `results/register.csv` with JSONL (one run per line) stored outside the repo under `$COREX_DATA_DIR/results/`; migrate the 59 existing rows; update `.gitignore` and README. | `corex_eval/submit.py`, `config.py` | No GitHub push in the submit path; old rows readable. |
| 1.3 | **Automatic run metadata**: exact model version string, snapshot ID where exposed, access date, temperature, prompt-template hash. | `corex_eval/submit.py`, experiment `config.yaml` schema | Submit refuses a run without them. |
| 1.4 | **Pin model snapshots.** No moving aliases such as `gpt-5.3-chat-latest`. | experiment configs | Every config names a dated snapshot. |

## Phase 2 — Silver standard

| # | Task | Notes |
|---|---|---|
| 2.1 | **Rebuild the silver set without the gold category hint** (`build_silver.py:123`, and the coded subject in the education prompt as far as it can be removed). | Costs one GPT pass over all cases. Expect more failed matches, since spells are then located by years only. |
| 2.2 | **Sensitivity check**: rerun two or three annotation models on old vs new silver and report the difference. | Quantifies how much the hint inflated scores. |
| 2.3 | **Language of inputs**: record per row whether the label came from `cv_english` or `cv_local`; decide whether to add a local-language annotation run. | Needed to interpret English BERT ≈ multilingual BERT. |

## Phase 3 — Decisions to write down before running

- **Model lineup**: current strongest closed and open-weight models from a general leaderboard; replace Gemini 2.5 Flash.
- **Country inclusion rule**: as specified by Tom; document it in the repo and list excluded countries in an appendix.

## Phase 4 — Pipeline and prompt work (all tuned on the dev split)

| # | Task |
|---|---|
| 4.1 | **Baselines for annotation**: majority class; TF-IDF + logistic regression; nearest-neighbour vote on the same embeddings without an LLM; fixed few-shot prompting for the generative models. |
| 4.2 | **Long CVs on local models**: chunk, extract per chunk, merge; verify against unchunked output on CVs that do fit. |
| 4.3 | **Birth-country prompt ablation**: with vs without the coded country in the prompt. |
| 4.4 | **Career-extraction prompt ablation**: with vs without simplified WG2 spell-boundary rules; inspect the 62-spell outlier. |
| 4.5 | **RAG-WebSearch ablation**: raw snippets without the summary step; alternative queries. |

## Phase 5 — Final runs (test split, once)

- Full matrix with the fixed lineup, including GPT on `uni_subject`.
- Confusion matrices for career sector and university subject, overall and per country.
- Report or drop the semantic-similarity metric.

## Phase 6 — Paper

- Reword title, abstract and introduction: collection is defined, not benchmarked.
- State the silver-standard change and the English-input caveat.
- Add intervals to all tables; add the baselines; add the stricter career metric.
- Note that models may recall facts about prominent politicians from training data.
- Fixes: `table_education_gen` included twice; `\label{fig: highest_degr}` should be `\ref`; `R_i` used for both recall and the retrieved set; "birth country" listed under both model groups in the atomic-extraction paragraph.

---

## Order and dependencies

- 0.1 blocks all of Phase 4. 1.1 blocks 0.3 in practice and all confusion matrices.
- 2.1 changes every annotation number, so it must precede Phase 5 and ideally 4.1.
- Phases 0 and 1 need no API calls and can start immediately.

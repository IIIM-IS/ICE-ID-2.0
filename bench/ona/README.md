# ONA (OpenNARS for Applications) matcher and official-split baseline reruns

Experiment code for the NARS comparison on ICE-ID and the eight DeepMatcher benchmarks. ONA scores each
record pair by answering `<{pair} --> match>?` from implications whose truth values are counted from
training pairs; the additive log-likelihood-ratio control (the published NARS/LLR matcher) and logistic
regression on the same judgments are the two controls.

## ONA runtime

- ONA is used unmodified: commit `f14cfce95ce9ffc2eb1a2ac52ac827bc25d528ee` of
  https://github.com/opennars/OpenNARS-for-Applications, built with its `build.sh`.
- Binary SHA-256 used for the reported runs: `d006f11b10e01c146e61c27dfb891ebd924f8c56433ea7886b83ff6cfb7a6187`
  (checked by `ona_matcher.verify_runtime()`). The judgment-set runs used a second build of the same commit
  (SHA-256 `04bec091e9a780497d25a68fb77881be1f60d2704f04d1f6da27b81afde16729`); on 112 archived batches
  (10,679 episodes) its output equals the first build's output byte for byte.
- Set `ONA_DIR=<path to the ONA checkout containing NAR>`. This is the only change from the original
  scripts, which had the checkout path hard-coded.

## Scripts

ONA matcher and runners:

| File | Purpose |
|---|---|
| `ona_matcher.py` | Judgment atoms, implications from training counts (f = w+/w, c = w/(w+1) capped at 0.99), episode text, answer parsing, batched NAR shell runs, closed-form NAL reference. |
| `ona_cache.py` | Distinct-episode cache: a pair's episode depends only on its judgment set, so each distinct episode is run once with a constant pair term. |
| `run_ona_full.py` | ICE-ID scoring for every seed and protocol (`iceid`), and per-pair reruns as fresh episodes (`fresh-check`). |
| `run_ona_external.py` | The eight DeepMatcher benchmarks on their official train/valid/test splits (Fodors-Zagats without `class`). |
| `run_ona_metrics.py` | ICE-ID metric tables; recomputes the additive control and checks it against its saved predictions. |
| `run_ona_summary.py` | `summary.md` / `summary.csv` of the full run. |
| `verify_ona_full.py` | Checks every ONA answer against the closed-form NAL derivation. |
| `run_ona_engine_comparison.py` | Bootstrap (fixed and reselected thresholds), tie, reliability, label-audit, threshold-rule, mover, birth-year-veto and cluster re-analyses of the saved scores. |
| `run_judgment_lr.py` | Logistic regression on ONA's judgments (the jointly fitted control). |
| `run_ona_setcount.py` | Judgment-set implications: one atom per training judgment set, counted from training pairs, revised by ONA with the single-judgment conclusions (`prepare`, `run`, `fresh-check`, `score`). |
| `run_ona_setcount_metrics.py`, `run_ona_setcount_summary.py` | Metrics and summary of the judgment-set runs. |

Official-split baseline reruns:

| File | Purpose |
|---|---|
| `run_official_baselines.py` | Fellegi-Sunter, Ditto, ZeroER and AnyMatch on the official DeepMatcher splits of seven benchmarks, through the Fodors-Zagats driver (`run_fodors_official.py`) re-targeted per dataset. AnyMatch leaves both DBLP domains out of training when either is the target, and may use activation checkpointing under a GPU-memory cap. Ditto may accumulate gradients at the same effective batch; ZeroER self-joins records of the test pairs. |
| `verify_official_baselines.py` | Recomputes every reported cell from the saved predictions and checks the test split, decision rule and hashes. |

## Expected workspace layout

The scripts were run from the experiment workspace (`label-repair/`) and import its modules by name. They
are published here for reference; to rerun them, place them next to these modules:

- `prepare_release` (paths, release hashes), `experiment_data` (pair sampling and features), `fitted_models`,
  `resource_limits` (memory gate), `run_experiments` (metrics, threshold on validation),
  `publication_evaluation` (full-node clusters), `publication_run`, `run_ona_pilot` (episode design),
  `audit_labels` (benchmark label audit), `run_matchgpt_external` (official-split loading);
- for the baselines: `run_fodors_official`, `run_hiergat_external`, `run_anymatch_transfer`,
  `prepare_anymatch_transfer`, and the upstream method code (`ditto_light.ditto` from Ditto, ZeroER's
  `blocking_functions` with `py_entitymatching`, the AnyMatch repository);
- `bench.models.nars` from this repository's `bench/` package (`vendor/bench` in the workspace).

Data and outputs are read from and written to `experiments/` under the workspace (`experiments/ona/full`,
`experiments/ona/setcount`, `experiments/completion/official-baselines`). Per-pair scores, ONA transcripts,
raw data and model weights are not included here.

## Results (`results/`)

| File | Content |
|---|---|
| `ona_full_summary.md`, `ona_full_summary.csv` | ONA and the additive control on ICE-ID (controlled pairs, candidate graphs, second assessment) and the external benchmarks. |
| `external_results.csv` | ONA and the additive control on the eight official splits (181-point threshold grid; `run_ona_engine_comparison.py` also applies the ICE-ID rule, the highest tied observed validation score). |
| `official_baselines.csv` | Fellegi-Sunter, Ditto and ZeroER on the seven official splits other than Fodors-Zagats (21 verified cells). |
| `setcount_summary.md`, `setcount_summary.csv` | ONA with judgment-set implications against single-judgment ONA, the additive control and logistic regression on the judgments. Movers there are true matches whose commune and county judgments both differ. |

# ONA replacement of the NARS/LLR matcher: full results

Results only (no paper or existing file edited). ONA = OpenNARS for Applications does the inference (one reset episode per distinct input; score = expectation of ONA's printed answer). LLR = the additive log-likelihood-ratio control with the identical judgments and the identical training counts, recomputed here on the same pairs (every LLR cell reproduces the published value; `iceid/tables/llr_reproduction.json`). Values: mean ± sample SD over seeds 131-133 where the paper uses three seeds.

## Runtime and settings

- ONA binary `<ONA_DIR>/NAR`, sha256 `d006f11b10e01c146e61c27dfb891ebd924f8c56433ea7886b83ff6cfb7a6187`, commit `f14cfce95ce9ffc2eb1a2ac52ac827bc25d528ee`, src clean: True, mode NAR shell (stdin/stdout)
- Design (pilot, unchanged): {"batch_size": 100, "cycles": 50, "form": "decomposition", "knowledge_scope": "present", "order": "sorted", "step_cycles": 3, "workers": 3}; implication truth f = w+/w (or 1-f for the negated form), c = w/(w+1) capped at 0.99; `*currenttime=1000000000` after each `*reset`; `*volume=0`.
- Episode cache: constant pair term `{px}`; ICE-ID 1,630,070 pair scorings from 4,034 distinct episodes (1,250 distinct judgment sets).

## ICE-ID headline cells

| Cell | ONA | NARS/LLR (additive control) | ONA - LLR |
|---|---|---|---|
| Controlled P | 0.930 | 0.942 | -0.012 |
| Controlled R | 0.995 | 0.994 | +0.001 |
| Controlled F1 | 0.962 ± 0.005 | 0.967 ± 0.006 | -0.006 |
| Controlled AUROC | 0.994 | 0.995 | -0.001 |
| Controlled AP | 0.988 | 0.990 | -0.002 |
| Controlled threshold (selection F1) | 0.2698 | 0.0208 | +0.2491 |
| Audit inherited F1 | 0.794 ± 0.093 | 0.841 ± 0.003 | -0.047 |
| Audit provisional F1 | 0.331 ± 0.021 | 0.317 ± 0.008 | +0.014 |
| Audit different linked (of 117) | 72 | 80 | -8 |
| Audit without 1816 F1 | 0.338 | 0.323 | +0.015 |
| Audit without 1816 different linked (of 113) | 70 | 78 | -8 |
| Audit source-qualified 135 F1 | 0.331 | 0.317 | +0.014 |
| Audit source-qualified different linked (of 116) | 72 | 80 | -8 |
| Candidate graph edge F1 | 0.918 ± 0.003 | 0.908 ± 0.011 | +0.010 |
| Candidate graph AP | 0.970 | 0.971 | -0.001 |
| Candidate graph B3 P | 0.992 | 0.984 | +0.009 |
| Candidate graph B3 R | 0.898 | 0.899 | -0.001 |
| Candidate graph B3 F1 | 0.943 ± 0.001 | 0.939 ± 0.003 | +0.003 |
| Candidate graph ARI | 0.828 | 0.810 | +0.018 |
| Candidate graph partition pair F1 | 0.828 ± 0.006 | 0.810 ± 0.018 | +0.018 |
| Candidate graph P@k | 0.874 | 0.873 | +0.001 |
| Candidate graph largest cluster | 6 | 7 | -1 |
| Controlled-trained graph edge F1 | 0.811 ± 0.010 | 0.808 ± 0.011 | +0.003 |
| Controlled-trained graph B3 F1 | 0.911 ± 0.002 | 0.909 ± 0.003 | +0.001 |
| Controlled-trained graph ARI | 0.667 | 0.650 | +0.017 |
| Locked controlled F1 | 0.968 ± 0.002 | 0.975 ± 0.009 | -0.008 |
| Locked edge F1 | 0.907 ± 0.002 | 0.894 ± 0.019 | +0.013 |
| Locked B3 F1 | 0.976 ± 0.000 | 0.973 ± 0.004 | +0.003 |
| Locked ARI | 0.851 | 0.818 | +0.033 |

## Ablations (controlled test; counts re-fitted on train without the category, threshold re-selected)

| Vocabulary | ONA F1 | ONA dF1 vs full | ONA AUROC | LLR F1 | LLR dF1 | LLR AUROC |
|---|---|---|---|---|---|---|
| Without names | 0.936 ± 0.003 | -0.026 | 0.989 | 0.935 ± 0.003 | -0.033 | 0.989 |
| Without geography | 0.974 ± 0.001 | 0.013 | 0.997 | 0.969 ± 0.007 | 0.002 | 0.997 |
| Without birth year | 0.891 ± 0.006 | -0.070 | 0.980 | 0.895 ± 0.006 | -0.072 | 0.981 |
| Without sex | 0.962 ± 0.005 | 0.000 | 0.994 | 0.967 ± 0.006 | 0.000 | 0.995 |
| Without census wave | 0.962 ± 0.005 | 0.000 | 0.994 | 0.967 ± 0.006 | 0.000 | 0.995 |

## Calibration (maps fit on the calibration partition, threshold on selection)

| Population | Map | ONA F1 | LLR F1 | ONA Brier | LLR Brier | ONA log loss | LLR log loss |
|---|---|---|---|---|---|---|---|
| Controlled pairs | raw | 0.962 | 0.967 | 0.0604 | 0.0482 | 0.2343 | 0.1995 |
| Controlled pairs | sigmoid | 0.962 | 0.967 | 0.0272 | 0.0243 | 0.0979 | 0.0882 |
| Controlled pairs | isotonic | 0.963 | 0.968 | 0.0218 | 0.0187 | 0.0957 | 0.0897 |
| Candidate graph (candidate-trained) | raw | 0.918 | 0.908 | 0.0129 | 0.0042 | 0.1026 | 0.0203 |
| Candidate graph (candidate-trained) | sigmoid | 0.918 | 0.908 | 0.0023 | 0.0024 | 0.0083 | 0.0078 |
| Candidate graph (candidate-trained) | isotonic | 0.916 | 0.911 | 0.0017 | 0.0017 | 0.0060 | 0.0060 |
| Candidate graph (controlled-trained) | raw | 0.811 | 0.808 | 0.0380 | 0.0150 | 0.2000 | 0.0742 |
| Candidate graph (controlled-trained) | sigmoid | 0.811 | 0.808 | 0.0052 | 0.0056 | 0.0219 | 0.0208 |
| Candidate graph (controlled-trained) | isotonic | 0.813 | 0.801 | 0.0045 | 0.0046 | 0.0166 | 0.0158 |

## Locked (fixed-model) calibration

| Population | Map | ONA Brier | LLR Brier | ONA log loss | LLR log loss |
|---|---|---|---|---|---|
| controlled | raw | 0.0584 | 0.0426 | 0.2283 | 0.1687 |
| controlled | sigmoid | 0.0228 | 0.0189 | 0.0748 | 0.0607 |
| controlled | isotonic | 0.0170 | 0.0138 | 0.0551 | 0.0528 |
| candidate | raw | 0.0132 | 0.0047 | 0.1033 | 0.0212 |
| candidate | sigmoid | 0.0019 | 0.0020 | 0.0076 | 0.0071 |
| candidate | isotonic | 0.0016 | 0.0016 | 0.0060 | 0.0064 |

## Threshold-policy sensitivity (controlled test)

| Policy | ONA F1 | LLR F1 | ONA mean threshold | LLR mean threshold |
|---|---|---|---|---|
| fixed_0.5 | 0.871 ± 0.001 | 0.917 ± 0.002 | 0.5000 | 0.5000 |
| training_f1 | 0.963 ± 0.004 | 0.937 ± 0.032 | 0.2919 | 0.1817 |
| selection_f1 | 0.962 ± 0.005 | 0.967 ± 0.006 | 0.2698 | 0.0208 |

## Component bootstrap (seed 131; 500 resamples, rng 571, identical resamples for all models)

| Population | Components | ONA F1 95% | LLR F1 95% | ONA - LLR paired 95% |
|---|---|---|---|---|
| locked_controlled | 307 | [0.9576, 0.9742] | [0.9800, 0.9906] | [-0.0256, -0.0128] |
| publication_test | 959 | [0.9474, 0.9643] | [0.9676, 0.9802] | [-0.0237, -0.0124] |

Every other model's interval and its delta versus LLR reproduce the saved uncertainty.json exactly; deltas versus ONA for every model are in `iceid/tables/uncertainty.json` (`reference_ona`).

## Table 5 (F1) and the external benchmarks on the official DeepMatcher splits

External: counts from the official train split, threshold maximising validation F1 (run_nars_full_eval.select_threshold_by_f1), official test scored once, metrics by the executed NARS evaluator. Fodors-Zagats without `class` (run_fodors_official split; the LLR rerun reproduces experiments/completion/fodors-official/nars exactly). The archived nars_full_eval.csv external cells used unretained sampled pairs and are not comparable.

| Dataset | ONA F1 | NARS/LLR F1 (same split) | ONA - LLR |
|---|---|---|---|
| ICE-ID (controlled, 3 seeds) | 0.962 | 0.967 | -0.006 |
| abt_buy | 0.3647 | 0.3559 | 0.0088 |
| amazon_google | 0.4162 | 0.4513 | -0.0351 |
| dblp_acm | 0.9680 | 0.9539 | 0.0141 |
| dblp_scholar | 0.8776 | 0.8673 | 0.0103 |
| itunes_amazon | 0.9200 | 1.0000 | -0.0800 |
| walmart_amazon | 0.7018 | 0.7120 | -0.0102 |
| beer | 0.7879 | 0.8000 | -0.0121 |
| fodors_zagats | 0.9268 | 0.9778 | -0.0510 |

`auc` is the evaluator's own (bench _roc_auc_score: stable-sort ranks, ties not averaged, so it depends on pair order whenever scores tie, as they do for both matchers here); `auc_tie_averaged` is sklearn roc_auc_score on the same scores.

ONA (nars_all_benchmarks layout):

| Dataset | precision | recall | f1 | accuracy | threshold | auc | auc_tie_averaged | ari_cc | ari_ag | p_at_k | r_at_k | test pairs/pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abt_buy | 0.4627 | 0.3010 | 0.3647 | 0.8873 | 0.2350 | 0.7095 | 0.7107 | 0.4377 | 0.6772 | 0.3447 | 0.3447 | 1916/206 |
| amazon_google | 0.3946 | 0.4402 | 0.4162 | 0.8740 | 0.2300 | 0.8232 | 0.8242 | 0.3732 | 0.6245 | 0.4017 | 0.4017 | 2293/234 |
| dblp_acm | 0.9837 | 0.9527 | 0.9680 | 0.9887 | 0.5100 | 0.9970 | 0.9970 | 0.9906 | 0.9929 | 0.9572 | 0.9572 | 2473/444 |
| dblp_scholar | 0.8554 | 0.9009 | 0.8776 | 0.9532 | 0.3250 | 0.9812 | 0.9811 | 0.7965 | 0.7907 | 0.8626 | 0.8626 | 5742/1070 |
| itunes_amazon | 1.0000 | 0.8519 | 0.9200 | 0.9633 | 0.5200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 109/27 |
| walmart_amazon | 0.8054 | 0.6218 | 0.7018 | 0.9502 | 0.1900 | 0.9089 | 0.9084 | 0.8820 | 0.9020 | 0.6632 | 0.6632 | 2049/193 |
| beer | 0.6842 | 0.9286 | 0.7879 | 0.9231 | 0.1900 | 0.9777 | 0.9773 | 0.8083 | 0.8083 | 0.7857 | 0.7857 | 91/14 |
| fodors_zagats | 1.0000 | 0.8636 | 0.9268 | 0.9841 | 0.4200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 189/22 |

NARS/LLR (nars_all_benchmarks layout):

| Dataset | precision | recall | f1 | accuracy | threshold | auc | auc_tie_averaged | ari_cc | ari_ag | p_at_k | r_at_k | test pairs/pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abt_buy | 0.4257 | 0.3058 | 0.3559 | 0.8810 | 0.2400 | 0.6983 | 0.6995 | 0.3942 | 0.6544 | 0.3447 | 0.3447 | 1916/206 |
| amazon_google | 0.4679 | 0.4359 | 0.4513 | 0.8918 | 0.2650 | 0.8291 | 0.8301 | 0.4793 | 0.6366 | 0.4573 | 0.4573 | 2293/234 |
| dblp_acm | 0.9295 | 0.9797 | 0.9539 | 0.9830 | 0.9200 | 0.9972 | 0.9972 | 0.8716 | 0.9775 | 0.9572 | 0.9572 | 2473/444 |
| dblp_scholar | 0.8235 | 0.9159 | 0.8673 | 0.9478 | 0.7100 | 0.9832 | 0.9831 | 0.6852 | 0.7521 | 0.8804 | 0.8804 | 5742/1070 |
| itunes_amazon | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 109/27 |
| walmart_amazon | 0.9483 | 0.5699 | 0.7120 | 0.9566 | 0.8950 | 0.9112 | 0.9108 | 0.9733 | 0.9733 | 0.6788 | 0.6788 | 2049/193 |
| beer | 0.7500 | 0.8571 | 0.8000 | 0.9341 | 0.9100 | 0.9824 | 0.9819 | 0.8531 | 0.8531 | 0.7857 | 0.7857 | 91/14 |
| fodors_zagats | 0.9565 | 1.0000 | 0.9778 | 0.9947 | 0.9300 | 1.0000 | 1.0000 | 0.9773 | 0.9773 | 1.0000 | 1.0000 | 189/22 |

## Anomalies (never replaced)

Anomaly = unanswered episode, or |ONA - clean NAL derivation| > 1e-5 in frequency, confidence or expectation.

| domain | distinct_episodes | answered | unanswered | deviation_gt_1e-5 | max_abs_dev_expectation | max_abs_dev_frequency | max_abs_dev_confidence |
|---|---|---|---|---|---|---|---|
| iceid | 4034 | 4034 | 0 | 0 | 6.405181824375994e-07 | 5.00571385786408e-07 | 4.996612915642018e-07 |
| external | 1245 | 1245 | 0 | 0 | 6.48880824183351e-07 | 4.98174279439656e-07 | 4.998445768400472e-07 |

Pair-level (each pair scored in each protocol counted once):

| domain | pairs | unanswered | anomalous |
|---|---|---|---|
| external | 29725 | 0 | 0 |
| iceid | 1630070 | 0 | 0 |

Cells not computed because of unanswered pairs: 0

## Cache equivalence

```json
{
 "all_identical": true,
 "comparison": "complete per-episode ONA output (every line from the episode's first output to its Answer line), per-pair term {pv<n>} and marker renamed to the constant ones; per-pair episodes reset-separated, 100 per NAR process as in the pilot, the first 64 of each domain also alone in their own NAR process",
 "sampling": {
  "external": {
   "answer_line_identical": 700,
   "batched_seconds": 89.39039659500122,
   "distinct_episode_extra": 100,
   "distinct_episodes_covered": 348,
   "episodes_in_store": 1245,
   "own_process_checked": 64,
   "own_process_identical": 64,
   "own_process_seconds": 10.615064144134521,
   "pairs": 700,
   "random_pairs": 600,
   "segment_identical": 700
  },
  "iceid": {
   "answer_line_identical": 1800,
   "batched_seconds": 592.4471497535706,
   "distinct_episode_extra": 300,
   "distinct_episodes_covered": 655,
   "episodes_in_store": 4034,
   "own_process_checked": 64,
   "own_process_identical": 64,
   "own_process_seconds": 29.5497727394104,
   "pairs": 1800,
   "random_pairs": 1500,
   "segment_identical": 1800
  }
 },
 "total_pairs": 2500
}
```

## Concept-memory saturation

```json
{
 "by_n_judgments": [
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 14,
   "min_concepts": 4,
   "n_judgments": 1,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 75,
   "min_concepts": 75,
   "n_judgments": 2,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 178,
   "min_concepts": 970,
   "n_judgments": 3,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 312,
   "min_concepts": 4296,
   "n_judgments": 4,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 259,
   "min_concepts": 6452,
   "n_judgments": 5,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 167,
   "min_concepts": 8449,
   "n_judgments": 6,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 88,
   "min_concepts": 10549,
   "n_judgments": 7,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 56,
   "min_concepts": 12693,
   "n_judgments": 8,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 57,
   "min_concepts": 14800,
   "n_judgments": 9,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 29,
   "min_concepts": 16384,
   "n_judgments": 10,
   "saturated": 29
  },
  {
   "anomalies": 0,
   "domain": "external",
   "episodes": 10,
   "min_concepts": 16384,
   "n_judgments": 11,
   "saturated": 10
  },
  {
   "anomalies": 0,
   "domain": "iceid",
   "episodes": 1,
   "min_concepts": 7129,
   "n_judgments": 5,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "iceid",
   "episodes": 10,
   "min_concepts": 9144,
   "n_judgments": 6,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "iceid",
   "episodes": 17,
   "min_concepts": 10876,
   "n_judgments": 7,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "iceid",
   "episodes": 30,
   "min_concepts": 12766,
   "n_judgments": 8,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "iceid",
   "episodes": 168,
   "min_concepts": 14086,
   "n_judgments": 9,
   "saturated": 0
  },
  {
   "anomalies": 0,
   "domain": "iceid",
   "episodes": 163,
   "min_concepts": 16136,
   "n_judgments": 10,
   "saturated": 162
  },
  {
   "anomalies": 0,
   "domain": "iceid",
   "episodes": 11,
   "min_concepts": 16384,
   "n_judgments": 11,
   "saturated": 11
  }
 ],
 "concepts_max": 16384,
 "note": "total concepts after the answer; *stats appended after the question",
 "summary": [
  {
   "answer_identical": 1245,
   "domain": "external",
   "episodes": 1245,
   "max_judgments": 11,
   "saturated": 39
  },
  {
   "answer_identical": 400,
   "domain": "iceid",
   "episodes": 400,
   "max_judgments": 11,
   "saturated": 173
  }
 ]
}
```

## Independent verification

```json
{
 "iceid_episodes": {
  "episodes": 4034,
  "transcript_answers": 4034,
  "expectation_or_line_mismatch": 0,
  "input_hash_mismatch": 0
 },
 "external_episodes": {
  "episodes": 1245,
  "transcript_answers": 1245,
  "expectation_or_line_mismatch": 0,
  "input_hash_mismatch": 0
 },
 "pair_scores_from_episodes": {
  "pairs": 1659795,
  "mismatches": 0
 },
 "independent_metrics": {
  "rows_checked": 35,
  "mismatches": []
 },
 "thresholds": {
  "checked": 35,
  "all_equal": true
 }
}
```

## ONA per-pair explanations (seed 131, the pairs of the current figures)

Records 560071 / 717045 (reference 1); ONA: `Answer: <{px} --> match>. creationTime=1000000055 Stamp=[2,16,18,15,20,17,14,19,12,13] Truth: frequency=0.394656, confidence=0.921491`; identical to the scored run: True; shares reproduce f (|diff| 4.8e-07) and c (|diff| 2.6e-08); expectation 0.402926 (LLR raw score 0.792133).

| Judgment | w+ | w | conclusion f | conclusion c | w = c/(1-c) | share | share x f |
|---|---|---|---|---|---|---|---|
| adjacent_census | 3532 | 10596 | 0.333333 | 0.356400 | 0.5538 | 0.0472 | 0.0157 |
| different_county | 500 | 8310 | 0.060168 | 0.708305 | 2.4282 | 0.2069 | 0.0124 |
| different_farm | 1530 | 9507 | 0.160934 | 0.564563 | 1.2965 | 0.1105 | 0.0178 |
| different_parish | 667 | 8536 | 0.078140 | 0.681476 | 2.1395 | 0.1823 | 0.0142 |
| same_birthyear | 2384 | 2519 | 0.946407 | 0.718251 | 2.5493 | 0.2172 | 0.2056 |
| same_district | 3782 | 4698 | 0.805023 | 0.519681 | 1.0820 | 0.0922 | 0.0742 |
| same_first_name | 3993 | 11993 | 0.332944 | 0.356816 | 0.5548 | 0.0473 | 0.0157 |
| same_nafn_norm | 3921 | 8229 | 0.476486 | 0.219775 | 0.2817 | 0.0240 | 0.0114 |
| same_patronym | 3923 | 8440 | 0.464810 | 0.229687 | 0.2982 | 0.0254 | 0.0118 |
| same_sex | 4000 | 11997 | 0.333417 | 0.356311 | 0.5535 | 0.0472 | 0.0157 |

Records 561138 / 690113 (reference 0); ONA: `Answer: <{px} --> match>. creationTime=1000000055 Stamp=[2,18,20,17,16,19,14,15,12,13] Truth: frequency=0.415035, confidence=0.945265`; identical to the scored run: True; shares reproduce f (|diff| 1.2e-07) and c (|diff| 1.9e-07); expectation 0.419686 (LLR raw score 0.621474).

| Judgment | w+ | w | conclusion f | conclusion c | w = c/(1-c) | share | share x f |
|---|---|---|---|---|---|---|---|
| adjacent_census | 3532 | 10596 | 0.333333 | 0.356400 | 0.5538 | 0.0321 | 0.0107 |
| different_birthyear | 0 | 6618 | 0.000000 | 0.801900 | 4.0480 | 0.2344 | 0.0000 |
| different_farm | 1530 | 9507 | 0.160934 | 0.564563 | 1.2965 | 0.0751 | 0.0121 |
| different_nafn_norm | 79 | 3771 | 0.020949 | 0.768653 | 3.3225 | 0.1924 | 0.0040 |
| same_county | 3495 | 3680 | 0.949728 | 0.723301 | 2.6140 | 0.1514 | 0.1438 |
| same_district | 3782 | 4698 | 0.805023 | 0.519681 | 1.0820 | 0.0626 | 0.0504 |
| same_first_name | 3993 | 11993 | 0.332944 | 0.356816 | 0.5548 | 0.0321 | 0.0107 |
| same_parish | 3328 | 3449 | 0.964917 | 0.746621 | 2.9467 | 0.1706 | 0.1646 |
| same_patronym | 3923 | 8440 | 0.464810 | 0.229687 | 0.2982 | 0.0173 | 0.0080 |
| same_sex | 4000 | 11997 | 0.333417 | 0.356311 | 0.5535 | 0.0321 | 0.0107 |

## Notes for integration

- ONA scores are coarse: every pair with the same judgment set gets the same answer, so scores tie heavily (ICE-ID 1,250 distinct judgment sets over 1.63M scorings; external 31-404 distinct episodes per dataset). The selection-F1 threshold rule and tie-averaged sklearn AUROC/AP handle ties; the external evaluator's `auc` does not (see above).
- ONA expectations sit in a narrow band (controlled selection-F1 thresholds 0.256-0.292 versus 0.016-0.028 for LLR); the raw ONA expectation is not a calibrated probability (raw Brier/log loss worse than LLR; after sigmoid/isotonic maps they are close).
- Concept memory (CONCEPTS_MAX 16,384) was full after 212 of the 213 diagnosed episodes with 10 or more judgments (none below 9 judgments); no answer in any domain deviates from the clean NAL derivation by more than 6.5e-7, so the saturation did not cost evidence here (`saturation/`).
- The training-F1 threshold policy scores the training pairs with knowledge counted on those same pairs (in-sample), as the LLR row does.
- Explanation pairs are the current figures' pairs (chosen by the LLR criteria). Under ONA (seed-131 threshold 0.261836) both are still linked: the reference-positive pair is a true positive and the reference-negative pair is still a false positive.
- External NARS/LLR on the official splits differs strongly from the archived nars_full_eval.csv cells (e.g. Abt-Buy 0.356 vs 0.648, iTunes-Amazon 1.000 vs 0.757); the archived cells used unretained sampled pairs. Both ONA and LLR rows here use identical splits and counts.
- Model names in the tables: `ona`, `ona_ablate_<category>`; LLR rows keep `published_nars` / `ablate_<category>` and equal the published rows.

## Runtime (seconds)

```json
{
 "iceid_ona_seconds": 1148.2073571681976,
 "iceid_wall_seconds_incl_judgments": 1169.8576374053955,
 "external_ona_wall_seconds_by_dataset (batch archive timestamps)": {
  "abt_buy": 4.2,
  "amazon_google": 8.3,
  "dblp_acm": 24.6,
  "dblp_scholar": 25.1,
  "itunes_amazon": 92.7,
  "walmart_amazon": 41.6,
  "beer": 11.7,
  "fodors_zagats": 48.7
 }
}
```

## Reproduce

```bash
cd label-repair
python source_alias.py run_ona_full.py iceid-inputs   # observed records, publication path
python run_ona_full.py iceid                          # re-fit counts, judgments, LLR checks, ONA episodes
python run_ona_metrics.py                             # all ICE-ID tables (LLR rows checked against the published ones)
python run_ona_external.py                            # 8 official DeepMatcher splits, LLR + ONA
python run_ona_full.py fresh-check                    # cached vs per-pair episodes
python run_ona_full.py saturation
python run_ona_full.py explain
python verify_ona_full.py                             # independent re-parse / recomputation
python run_ona_summary.py
```

## Files

- `iceid/tables/*.csv`: same columns as the published files; ONA rows model `ona` / `ona_ablate_<category>`, LLR rows the original names (metrics, audit_source_timing, threshold_sensitivity, calibration_metrics, graph_metrics, ablations, candidate_graph_metrics, candidate_ranking, candidate_calibration_metrics, locked_metrics, locked_calibration_metrics, knowledge, uncertainty.json, llr_reproduction.json)
- `iceid/pairs/seed_<s>/*.csv.gz`: per-pair id1, id2, label, knowledge base, episode_id, ona_score, llr_score, anomaly (the pair -> episode map)
- `episodes/<domain>/episodes.csv.gz` (one row per distinct input: answer line, f, c, NAL reference, deviation, anomaly), `episodes/<domain>/batches/*.nal.gz|*.out.gz` (every NAR process input/output)
- `external/<dataset>/{result.json,valid_scores.csv.gz,test_scores.csv.gz}`, `external/results.csv`
- `cache_check/`, `saturation/`, `explanations/`, `verification.json`, `summary.csv`

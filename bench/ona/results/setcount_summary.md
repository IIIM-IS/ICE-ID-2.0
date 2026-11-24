# ONA counting evidence for whole judgment sets (ICE-ID): results

## Design used, and its limits

- Knowledge: the unchanged per-judgment implications plus one implication per distinct training judgment set, by the same NAL rule (`om.implications_from_counts`: w = training pairs whose set is exactly S, w+ = matches, f = w+/w, c = min(w/(w+1), 0.99), decomposition form). Counts come from training pairs only (controlled train for the controlled/publication-graph knowledge; candidate-matched training pairs for the candidate knowledge). The set counts reproduce every per-judgment w/w+ as marginals, and no scored pair is a training pair (`inputs/prepare.json`).
- The set is one atom `set_<id>`, not one compound term (probes in `probes/`, ONA f14cfce95ce9). ONA's parser is binary: `(&, a, b, c, d)` is read as `(a & b)`. A term is a 64-slot heap: a 9-judgment intersection in an implication aborts ONA ("COMPOUND_TERM_SIZE_MAX too small"), and an 8-judgment one parses but is never matched. ICE-ID sets have 5-11 judgments.
- Episode (`revision`): the pilot episode (sorted implication-then-observation, 3 cycles each, 50 cycles, then the question) with `<{px} --> set_<id>>.` as one more observation, plus its implication if the set occurred in training. Unseen sets: no set implication, so ONA answers from the per-judgment conclusions alone. Every input has its own stamp, so ONA revises the set conclusion with the per-judgment ones (Inference_RevisionAndChoice). Nothing is chosen in Python.
- Choice (one observation; judgments by definition `<set_<id> --> j>`, so all conclusions share a stamp and ONA keeps the most confident one) was piloted and dropped. In unmodified ONA the derived observations lose priority before the third inference step: 85/120 pilot episodes unanswered with set knowledge, 114/120 without (`runs/choice_pilot`, `runs/choice_noset_pilot`). Revision pilot: 120/120 answered and faithful.
- Desktop NAR built from `git archive f14cfce95ce9` (sha256 04bec091...); laptop NAR sha256 d006f11b.... The desktop output equals the laptop output byte for byte on all 112 archived batches (10,679 episodes; `repro/report.json`).
- Out of scope: external benchmarks (stopped at the coordinator's request).

Every ONA score is the expectation of ONA's printed answer.

## candidate_graph

| Seed | Model | P | R | F1 | AUROC | ARI | mover recall (n) |
|---|---|---|---|---|---|---|---|
| 131 | ONA, set + per-judgment (revision) | 0.9695 | 0.9837 | 0.9766 | 0.9998 | 0.8901 | 0.952 (231) |
| 131 | ONA, per-judgment only | 0.9703 | 0.8640 | 0.9141 | 0.9994 | 0.8228 | 0.048 (231) |
| 131 | Additive NARS/LLR control | 0.9703 | 0.8640 | 0.9141 | 0.9994 | 0.8228 | 0.048 (231) |
| 131 | LR on ONA's judgments | 0.9910 | 0.9893 | 0.9902 | 0.9999 | 0.9127 | 0.974 (231) |
| 132 | ONA, set + per-judgment (revision) | 0.9655 | 0.9863 | 0.9758 | 0.9998 | 0.8933 | 0.945 (238) |
| 132 | ONA, per-judgment only | 0.9798 | 0.8676 | 0.9203 | 0.9994 | 0.8354 | 0.025 (238) |
| 132 | Additive NARS/LLR control | 0.9163 | 0.8744 | 0.8949 | 0.9995 | 0.7901 | 0.034 (238) |
| 132 | LR on ONA's judgments | 0.9762 | 0.9911 | 0.9836 | 0.9999 | 0.9071 | 0.962 (238) |
| 133 | ONA, set + per-judgment (revision) | 0.9724 | 0.9807 | 0.9765 | 0.9997 | 0.8833 | 0.951 (224) |
| 133 | ONA, per-judgment only | 0.9863 | 0.8618 | 0.9199 | 0.9993 | 0.8267 | 0.045 (224) |
| 133 | Additive NARS/LLR control | 0.9743 | 0.8618 | 0.9146 | 0.9994 | 0.8181 | 0.045 (224) |
| 133 | LR on ONA's judgments | 0.9803 | 0.9886 | 0.9844 | 0.9999 | 0.8966 | 0.964 (224) |

Three-seed F1 mean ± SD: ONA, set + per-judgment (revision) 0.9763 ± 0.0004; ONA, per-judgment only 0.9181 ± 0.0034; Additive NARS/LLR control 0.9079 ± 0.0113; LR on ONA's judgments 0.9861 ± 0.0036

## controlled

| Seed | Model | P | R | F1 | AUROC | ARI | mover recall (n) |
|---|---|---|---|---|---|---|---|
| 131 | ONA, set + per-judgment (revision) | 0.9249 | 0.9958 | 0.9591 | 0.9943 |  | 0.968 (156) |
| 131 | ONA, per-judgment only | 0.9192 | 0.9958 | 0.9560 | 0.9932 |  | 0.968 (156) |
| 131 | Additive NARS/LLR control | 0.9537 | 0.9950 | 0.9739 | 0.9958 |  | 0.962 (156) |
| 131 | LR on ONA's judgments | 0.9787 | 0.9958 | 0.9872 | 0.9987 |  | 0.968 (156) |
| 132 | ONA, set + per-judgment (revision) | 0.9357 | 0.9950 | 0.9645 | 0.9957 |  | 0.965 (142) |
| 132 | ONA, per-judgment only | 0.9350 | 0.9950 | 0.9641 | 0.9946 |  | 0.965 (142) |
| 132 | Additive NARS/LLR control | 0.9342 | 0.9942 | 0.9633 | 0.9953 |  | 0.958 (142) |
| 132 | LR on ONA's judgments | 0.9811 | 0.9950 | 0.9880 | 0.9992 |  | 0.965 (142) |
| 133 | ONA, set + per-judgment (revision) | 0.9439 | 0.9950 | 0.9688 | 0.9957 |  | 0.969 (162) |
| 133 | ONA, per-judgment only | 0.9365 | 0.9950 | 0.9648 | 0.9948 |  | 0.969 (162) |
| 133 | Additive NARS/LLR control | 0.9386 | 0.9933 | 0.9652 | 0.9952 |  | 0.957 (162) |
| 133 | LR on ONA's judgments | 0.9795 | 0.9950 | 0.9872 | 0.9988 |  | 0.969 (162) |

Three-seed F1 mean ± SD: ONA, set + per-judgment (revision) 0.9641 ± 0.0049; ONA, per-judgment only 0.9616 ± 0.0049; Additive NARS/LLR control 0.9674 ± 0.0057; LR on ONA's judgments 0.9875 ± 0.0005

## locked_candidate

| Seed | Model | P | R | F1 | AUROC | ARI | mover recall (n) |
|---|---|---|---|---|---|---|---|
| 131 | ONA, set + per-judgment (revision) | 0.9613 | 0.9756 | 0.9684 | 0.9995 | 0.9126 | 0.938 (97) |
| 131 | ONA, per-judgment only | 0.9648 | 0.8525 | 0.9052 | 0.9991 | 0.8477 | 0.021 (97) |
| 131 | Additive NARS/LLR control | 0.9648 | 0.8525 | 0.9052 | 0.9991 | 0.8477 | 0.021 (97) |
| 131 | LR on ONA's judgments | 0.9864 | 0.9797 | 0.9830 | 0.9998 | 0.9368 | 0.948 (97) |
| 132 | ONA, set + per-judgment (revision) | 0.9615 | 0.9797 | 0.9705 | 0.9995 | 0.9145 | 0.938 (97) |
| 132 | ONA, per-judgment only | 0.9752 | 0.8498 | 0.9082 | 0.9991 | 0.8529 | 0.021 (97) |
| 132 | Additive NARS/LLR control | 0.8811 | 0.8620 | 0.8714 | 0.9992 | 0.7581 | 0.021 (97) |
| 132 | LR on ONA's judgments | 0.9668 | 0.9838 | 0.9752 | 0.9999 | 0.9215 | 0.948 (97) |
| 133 | ONA, set + per-judgment (revision) | 0.9613 | 0.9756 | 0.9684 | 0.9995 | 0.9126 | 0.938 (97) |
| 133 | ONA, per-judgment only | 0.9752 | 0.8498 | 0.9082 | 0.9991 | 0.8529 | 0.021 (97) |
| 133 | Additive NARS/LLR control | 0.9648 | 0.8525 | 0.9052 | 0.9991 | 0.8477 | 0.021 (97) |
| 133 | LR on ONA's judgments | 0.9732 | 0.9824 | 0.9778 | 0.9998 | 0.9240 | 0.948 (97) |

Three-seed F1 mean ± SD: ONA, set + per-judgment (revision) 0.9691 ± 0.0012; ONA, per-judgment only 0.9072 ± 0.0017; Additive NARS/LLR control 0.8939 ± 0.0195; LR on ONA's judgments 0.9787 ± 0.0040

## locked_controlled

| Seed | Model | P | R | F1 | AUROC | ARI | mover recall (n) |
|---|---|---|---|---|---|---|---|
| 131 | ONA, set + per-judgment (revision) | 0.9455 | 0.9967 | 0.9704 | 0.9967 |  | 0.979 (144) |
| 131 | ONA, per-judgment only | 0.9380 | 0.9967 | 0.9665 | 0.9960 |  | 0.979 (144) |
| 131 | Additive NARS/LLR control | 0.9755 | 0.9958 | 0.9856 | 0.9982 |  | 0.972 (144) |
| 131 | LR on ONA's judgments | 0.9852 | 0.9975 | 0.9913 | 0.9995 |  |  |
| 132 | ONA, set + per-judgment (revision) | 0.9455 | 0.9967 | 0.9704 | 0.9966 |  | 0.979 (144) |
| 132 | ONA, per-judgment only | 0.9447 | 0.9967 | 0.9700 | 0.9960 |  | 0.979 (144) |
| 132 | Additive NARS/LLR control | 0.9440 | 0.9975 | 0.9700 | 0.9963 |  | 0.979 (144) |
| 132 | LR on ONA's judgments | 0.9860 | 0.9975 | 0.9917 | 0.9996 |  |  |
| 133 | ONA, set + per-judgment (revision) | 0.9455 | 0.9967 | 0.9704 | 0.9966 |  | 0.979 (144) |
| 133 | ONA, per-judgment only | 0.9373 | 0.9967 | 0.9661 | 0.9960 |  | 0.979 (144) |
| 133 | Additive NARS/LLR control | 0.9440 | 0.9975 | 0.9700 | 0.9963 |  | 0.979 (144) |
| 133 | LR on ONA's judgments | 0.9917 | 0.9975 | 0.9946 | 0.9995 |  |  |

Three-seed F1 mean ± SD: ONA, set + per-judgment (revision) 0.9704 ± 0.0000; ONA, per-judgment only 0.9675 ± 0.0022; Additive NARS/LLR control 0.9752 ± 0.0090; LR on ONA's judgments 0.9925 ± 0.0018

## publication_graph

| Seed | Model | P | R | F1 | AUROC | ARI | mover recall (n) |
|---|---|---|---|---|---|---|---|
| 131 | ONA, set + per-judgment (revision) | 0.9915 | 0.7197 | 0.8340 | 0.9929 | 0.7413 | 0.000 (231) |
| 131 | ONA, per-judgment only | 0.8544 | 0.7618 | 0.8055 | 0.9926 | 0.6661 | 0.022 (231) |
| 131 | Additive NARS/LLR control | 0.8380 | 0.7612 | 0.7978 | 0.9955 | 0.6347 | 0.022 (231) |
| 131 | LR on ONA's judgments | 0.2065 | 0.3152 | 0.2495 | 0.9194 | 0.0148 | 0.009 (231) |
| 132 | ONA, set + per-judgment (revision) | 0.9716 | 0.7725 | 0.8607 | 0.9930 | 0.7732 | 0.000 (238) |
| 132 | ONA, per-judgment only | 0.8740 | 0.7767 | 0.8225 | 0.9915 | 0.7004 | 0.013 (238) |
| 132 | Additive NARS/LLR control | 0.8651 | 0.7782 | 0.8194 | 0.9951 | 0.6757 | 0.013 (238) |
| 132 | LR on ONA's judgments | 0.1138 | 0.7551 | 0.1977 | 0.8622 | 0.0002 | 0.029 (238) |
| 133 | ONA, set + per-judgment (revision) | 0.9629 | 0.7378 | 0.8354 | 0.9917 | 0.7365 | 0.000 (224) |
| 133 | ONA, per-judgment only | 0.8393 | 0.7753 | 0.8060 | 0.9909 | 0.6338 | 0.036 (224) |
| 133 | Additive NARS/LLR control | 0.8400 | 0.7765 | 0.8070 | 0.9944 | 0.6382 | 0.036 (224) |
| 133 | LR on ONA's judgments | 0.2254 | 0.3333 | 0.2689 | 0.9274 | 0.0276 | 0.018 (224) |

Three-seed F1 mean ± SD: ONA, set + per-judgment (revision) 0.8434 ± 0.0150; ONA, per-judgment only 0.8113 ± 0.0097; Additive NARS/LLR control 0.8080 ± 0.0108; LR on ONA's judgments 0.2387 ± 0.0368

## Answer kinds (test / locked pairs)

| seed | population | design | pairs | seen_set_share | answer_mixed | answer_judgment | set_evidence_share_mean_seen | set_evidence_share_max |
|---|---|---|---|---|---|---|---|---|
| 131 | controlled | revision | 3600 | 0.983 | 3539 | 61 | 0.171 | 0.270 |
| 131 | publication_graph | revision | 122835 | 0.335 | 41113 | 81722 | 0.160 | 0.260 |
| 131 | candidate_graph | revision | 122835 | 0.973 | 119561 | 3274 | 0.151 | 0.217 |
| 131 | locked_controlled | revision | 3600 | 0.980 | 3527 | 73 | 0.167 | 0.270 |
| 131 | locked_candidate | revision | 73252 | 0.962 | 70450 | 2802 | 0.149 | 0.217 |
| 132 | controlled | revision | 3600 | 0.974 | 3506 | 94 | 0.173 | 0.258 |
| 132 | publication_graph | revision | 124509 | 0.333 | 41476 | 83033 | 0.157 | 0.258 |
| 132 | candidate_graph | revision | 124509 | 0.982 | 122278 | 2231 | 0.150 | 0.220 |
| 132 | locked_controlled | revision | 3600 | 0.979 | 3525 | 75 | 0.168 | 0.258 |
| 132 | locked_candidate | revision | 73252 | 0.970 | 71027 | 2225 | 0.147 | 0.220 |
| 133 | controlled | revision | 3600 | 0.979 | 3524 | 76 | 0.169 | 0.262 |
| 133 | publication_graph | revision | 119781 | 0.350 | 41975 | 77806 | 0.159 | 0.262 |
| 133 | candidate_graph | revision | 119781 | 0.974 | 116720 | 3061 | 0.154 | 0.238 |
| 133 | locked_controlled | revision | 3600 | 0.980 | 3528 | 72 | 0.167 | 0.262 |
| 133 | locked_candidate | revision | 73252 | 0.954 | 69878 | 3374 | 0.152 | 0.238 |

answer_mixed = the answer stamp names both the set implication and per-judgment implications (ONA revised them); answer_judgment = only per-judgment implications named (unseen set, or the set's ids truncated from the 10-id stamp). set_evidence_share = the set conclusion's evidence weight w = c/(1-c) over the sum in the clean NAL derivation, which ONA's answer reproduces (max deviation below).

## Episodes, coverage and cache check

```json
[
 {
  "answer_kinds": {
   "judgment": 1802,
   "mixed": 640
  },
  "answered": 2442,
  "design": "revision",
  "dev_gt_1e-5": 0,
  "episodes": 2442,
  "fresh_check": {
   "all_identical": true,
   "answer_line_identical": 400,
   "batched_seconds": 116.38411402702332,
   "comparison": "complete per-episode ONA output, per-pair term {pv<n>} and marker renamed to the constant ones; 100 per NAR process; the first 64 also alone in their own NAR process",
   "design": "revision",
   "distinct_episode_extra": 100,
   "distinct_episodes_covered": 207,
   "episodes_in_store": 2442,
   "own_process_checked": 64,
   "own_process_identical": 64,
   "own_process_seconds": 22.344974517822266,
   "pairs": 400,
   "random_pairs": 300,
   "runtime_sha256": "04bec091e9a780497d25a68fb77881be1f60d2704f04d1f6da27b81afde16729",
   "segment_identical": 400
  },
  "max_dev_reference": 5.005924205159218e-07,
  "ona_seconds": 453.19580817222595,
  "tied_references": 0,
  "unanswered": 0,
  "workers": 8
 }
]
```

Single-judgment ONA and LLR rows recomputed here equal the saved tables (max |diff| 0.0e+00); missing cells: none.


## Transcript example (seed 131, controlled knowledge, set 1; `probes/example_set1_seed131.{nal,out}`, volume 100, answer identical to the scored one)

```
Derived: <{px} --> match>. Priority=0.030626 Stamp=[2,1] Truth: frequency=0.333333, confidence=0.356400
...
Revised: <{px} --> match>. Priority=0.030626 Stamp=[2,20,16,19,12,15,4,11,10,3] Truth: frequency=0.860566, confidence=0.927941
Derived: <{px} --> (set_1 & match)>. Priority=0.283239 Stamp=[22,21] Truth: frequency=0.998975, confidence=0.890087
Revised: <{px} --> match>. Priority=0.098529 Stamp=[2,22,20,21,16,19,12,15,4,11] Truth: frequency=0.893410, confidence=0.944084
Revised: <{px} --> match>. Priority=0.029456 Stamp=[2,18,22,17,20,21,16,19,12,15] Truth: frequency=0.885972, confidence=0.945001
Revised: <{px} --> match>. Priority=0.029265 Stamp=[2,14,18,13,22,17,20,21,16,19] Truth: frequency=0.879367, confidence=0.945840
Answer: <{px} --> match>. creationTime=1000000058 Stamp=[2,14,18,13,22,17,20,21,16,19] Truth: frequency=0.879367, confidence=0.945840
```
The set conclusion (f 0.999) is revised in as one conclusion among 11 (about 17% of the evidence weight).

## Reproduce

```bash
cd label-repair
python run_ona_setcount.py prepare                      # laptop: set counts, knowledge, unlabelled desktop inputs
# desktop (<ONA_SETCOUNT_ROOT>: ona-src/ built by build.sh from git archive f14cfce95ce9; code/; inputs/):
python3 code/repro_check.py                             # desktop NAR vs archived laptop NAR output
systemd-run --user -p MemoryMax=16G python3 run_ona_setcount.py run --design revision --workers 8
systemd-run --user -p MemoryMax=16G python3 run_ona_setcount.py fresh-check --design revision
# laptop, after syncing runs/ back:
python run_ona_setcount.py score && python run_ona_setcount_summary.py
```

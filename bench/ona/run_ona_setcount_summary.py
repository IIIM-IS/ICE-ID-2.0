"""summary.csv / summary.md of the set-count ONA run from experiments/ona/setcount/tables (results only)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "experiments/ona/setcount"
NAMES = {"ona_set_revision": "ONA, set + per-judgment (revision)", "ona": "ONA, per-judgment only",
         "published_nars": "Additive NARS/LLR control", "lr_judgments": "LR on ONA's judgments"}
ORDER = list(NAMES)


def main():
    t = OUT / "tables"
    m = pd.read_csv(t / "metrics.csv")
    m = m[m.model.isin(ORDER)]
    cols = ["population", "seed", "model", "precision", "recall", "f1", "auc", "threshold", "ari"]
    s = m[[c for c in cols if c in m]].copy()
    s["order"] = s.model.map(ORDER.index)
    s = s.sort_values(["population", "seed", "order"]).drop(columns="order")
    mv = pd.read_csv(t / "movers.csv")
    s = s.merge(mv[["seed", "population", "model", "movers", "mover_recall"]], on=["seed", "population", "model"], how="left")
    s.to_csv(OUT / "summary.csv", index=False)
    kinds = pd.read_csv(t / "answer_kinds.csv")
    eps = json.loads((t / "episodes_summary.json").read_text())
    checks = json.loads((t / "checks.json").read_text())
    lines = ["# ONA counting evidence for whole judgment sets (ICE-ID): results", "",
             "Every ONA score is the expectation of ONA's printed answer; see run_ona_setcount.py for the design.", ""]
    for pop in s.population.unique():
        lines += [f"## {pop}", "", "| Seed | Model | P | R | F1 | AUROC | ARI | mover recall (n) |", "|---|---|---|---|---|---|---|---|"]
        for r in s[s.population.eq(pop)].itertuples(index=False):
            ari = "" if pd.isna(getattr(r, "ari", float("nan"))) else f"{r.ari:.4f}"
            mvr = "" if pd.isna(r.mover_recall) else f"{r.mover_recall:.3f} ({int(r.movers)})"
            lines.append(f"| {r.seed} | {NAMES[r.model]} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.auc:.4f} | {ari} | {mvr} |")
        mean = s[s.population.eq(pop)].groupby("model").f1.agg(["mean", "std"])
        lines += ["", "Three-seed F1 mean ± SD: " + "; ".join(f"{NAMES[k]} {mean.loc[k, 'mean']:.4f} ± {mean.loc[k, 'std']:.4f}"
                                                         for k in ORDER if k in mean.index), ""]
    table = ["| " + " | ".join(kinds.columns) + " |", "|" + "---|" * len(kinds.columns)]
    table += ["| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |"
              for r in kinds.itertuples(index=False)]
    lines += ["## Answer kinds (test / locked pairs)", "", *table, "",
              "answer_mixed = the answer stamp names both the set implication and per-judgment implications (ONA revised them); "
              "answer_judgment = only per-judgment implications named (unseen set, or the set's ids truncated from the 10-id stamp). "
              "set_evidence_share = the set conclusion's evidence weight w = c/(1-c) over the sum in the clean NAL derivation, "
              "which ONA's answer reproduces (max deviation below).", "",
              "## Episodes, coverage and cache check", "", "```json", json.dumps(eps, indent=1, default=str), "```", "",
              f"Single-judgment ONA and LLR rows recomputed here equal the saved tables (max |diff| "
              f"{max(c['max_abs_difference'] for c in checks['single_engine_rows_equal_saved']):.1e}); missing cells: "
              f"{checks['missing_cells'] or 'none'}.", ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()

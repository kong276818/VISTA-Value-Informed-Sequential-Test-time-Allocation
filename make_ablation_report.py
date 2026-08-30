#!/usr/bin/env python3
"""
Generate analysis report, LaTeX tables, and figures for the 20-ablation campaign.

Usage:
  cd /home/sclab/paper2
  python make_ablation_report.py [--no-figures]

Outputs:
  tables/ablations/   — LaTeX .tex files
  figures/ablations/  — PNG figures
  results/ablations/campaign_report.md — human-readable report
"""

import argparse, json, math, sys
from pathlib import Path

ROOT      = Path(__file__).parent
RES_DIR   = ROOT / "results" / "ablations"
TAB_DIR   = ROOT / "tables"  / "ablations"
FIG_DIR   = ROOT / "figures" / "ablations"
REPORT_MD = RES_DIR / "campaign_report.md"

for d in (TAB_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

CELLS = [
    ("m1_ds1", "Qwen3-8B",  "MMLU-Pro"),
    ("m1_ds2", "Qwen3-8B",  "ARC-Chall"),
    ("m1_ds3", "Qwen3-8B",  "MedMCQA"),
    ("m1_ds4", "Qwen3-8B",  "MedQA"),
    ("m2_ds1", "DeepSeek-R1-Llama", "MMLU-Pro"),
    ("m2_ds2", "DeepSeek-R1-Llama", "ARC-Chall"),
    ("m2_ds3", "DeepSeek-R1-Llama", "MedMCQA"),
    ("m2_ds4", "DeepSeek-R1-Llama", "MedQA"),
]
BUDGETS = [256, 512, 1024, 2048, 4096, 8192]


def load(fname: str) -> dict | None:
    p = RES_DIR / fname
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _fmt(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return format(v, fmt)


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _latex_esc(s: str) -> str:
    return str(s).replace("_", r"\_")


def latex_table(headers, rows, caption="", label=""):
    col_fmt = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A01 — IPW vs Naive table
# ---------------------------------------------------------------------------

def report_a01(d: dict) -> str:
    lines = ["\n## A01 — IPW vs Naive\n"]
    for cell_key, cell_d in d.items():
        if cell_key.startswith("_"): continue
        ds_name = cell_d.get("_meta", {}).get("ds_name", cell_key)
        rows_data = cell_d.get("rho_sweep", [])
        if not rows_data: continue
        frac_nat = rows_data[0].get("fraction_naturally_continued", float("nan"))
        lines.append(f"\n**{cell_key}** ({ds_name}, frac_natural_to_ref_b={frac_nat:.3f})")
        headers = ["rho", "IPW bias", "Naive bias", "Wrong-best IPW", "Wrong-best Naive"]
        rows = []
        for r in rows_data:
            rows.append([
                r["rho"],
                _fmt(r.get("ipw_mean_bias"), ".5f"),
                _fmt(r.get("naive_mean_bias"), ".5f"),
                _fmt(r.get("wrong_best_ipw_rate"), ".3f"),
                _fmt(r.get("wrong_best_naive_rate"), ".3f"),
            ])
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")

    # LaTeX table — all 8 cells at rho=0.20
    all_cells = [k for k in d if not k.startswith("_") and "rho_sweep" in d[k]]
    if all_cells:
        headers_tex = ["Cell", r"$f_{\text{nat}}$", "Naive bias", "IPW bias",
                       r"$\epsilon_{\text{naive}}$", r"$\epsilon_{\text{IPW}}$"]
        rows_tex = []
        for cell_key in sorted(all_cells):
            cd = d[cell_key]
            rsweep = cd.get("rho_sweep", [])
            # Use rho=0.20 row
            r = next((x for x in rsweep if abs(x["rho"] - 0.20) < 0.01), rsweep[2] if len(rsweep) > 2 else None)
            if r is None: continue
            frac_nat = r.get("fraction_naturally_continued", float("nan"))
            rows_tex.append([
                cell_key.replace("_", r"\_"),
                f"{frac_nat:.3f}",
                _fmt(r.get("naive_mean_bias"), ".5f"),
                _fmt(r.get("ipw_mean_bias"), ".5f"),
                _fmt(r.get("wrong_best_naive_rate"), ".3f"),
                _fmt(r.get("wrong_best_ipw_rate"), ".3f"),
            ])
        tex = latex_table(
            headers_tex, rows_tex,
            caption=r"IPW vs.\ naive risk estimation at $\rho=0.20$ across all cells. "
                    r"$f_{\text{nat}}$ = fraction of items that naturally continue to reference budget (4096 tokens). "
                    r"$\epsilon$ = wrong-best-budget selection rate ($n=200$ permutations).",
            label="tab:a01_ipw_naive",
        )
        (TAB_DIR / "a01_ipw_naive.tex").write_text(tex)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A02 — Counterfactual Missingness table
# ---------------------------------------------------------------------------

def report_a02(d: dict) -> str:
    lines = ["\n## A02 — Counterfactual Missingness\n"]
    headers = ["Cell", "Dataset", "FullInfo Brier", "ObsOnly bias", "NaiveAudit bias", "IPW bias"]
    rows = []
    for cell_key, cd in d.items():
        if cell_key.startswith("_"): continue
        ds_name = cd.get("_meta", {}).get("ds_name", cell_key)
        rows.append([
            _latex_esc(cell_key),
            ds_name,
            _fmt(cd.get("full_info_brier"), ".4f"),
            _fmt(cd.get("observed_only", {}).get("mean_bias"), ".5f"),
            _fmt(cd.get("naive_audited", {}).get("mean_bias"), ".5f"),
            _fmt(cd.get("ipw", {}).get("mean_bias"), ".6f"),
        ])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    # LaTeX
    tex_rows = [[r[0], r[2], r[3], r[4], r[5]] for r in rows]
    tex = latex_table(
        ["Cell", r"$\mathcal{R}^\star$", "Obs-only bias", "Naive audit bias", "IPW bias"],
        tex_rows,
        caption=r"Counterfactual missingness: bias in $\mathcal{R}^\star$ estimation under realistic stopping ($\tau=0.85$, $\rho=0.20$).",
        label="tab:a02_missingness",
    )
    (TAB_DIR / "a02_missingness.tex").write_text(tex)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A03 — CS vs Greedy
# ---------------------------------------------------------------------------

def report_a03(d: dict) -> str:
    lines = ["\n## A03 — CS vs Greedy\n"]
    headers = ["Cell", "Dataset", "CS FPR", "Greedy FPR", "Oracle best B"]
    rows = []
    for cell_key, cd in d.items():
        if cell_key.startswith("_"): continue
        ds_name = cd.get("_meta", {}).get("ds_name", cell_key)
        rows.append([
            _latex_esc(cell_key), ds_name,
            _fmt(cd.get("cs_false_positive_rate"), ".4f"),
            _fmt(cd.get("greedy_false_positive_rate"), ".4f"),
            str(cd.get("oracle_best", "?")),
        ])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    tex = latex_table(
        ["Cell", "Dataset", r"CS FPR $\downarrow$", r"Greedy FPR $\downarrow$", "Oracle $b^\star$"],
        rows,
        caption=r"Anytime-valid CS vs.\ greedy monitoring: false-positive policy-switch rate ($\rho=0.20$, $\alpha=0.05$, 200 stream permutations).",
        label="tab:a03_cs_greedy",
    )
    (TAB_DIR / "a03_cs_greedy.tex").write_text(tex)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A05 — Objective Divergence
# ---------------------------------------------------------------------------

def report_a05(d: dict) -> str:
    lines = ["\n## A05 — Objective Divergence (Acc vs Brier vs NLL)\n"]
    headers = ["Cell", "Dataset", "Best-Acc B", "Best-Brier B", "Best-NLL B", "Acc≠Brier?"]
    rows = []
    for cell_key, cd in d.items():
        if cell_key.startswith("_"): continue
        ds_name = cd.get("_meta", {}).get("ds_name", cell_key)
        rows.append([
            _latex_esc(cell_key), ds_name,
            str(cd.get("best_budget_accuracy", "?")),
            str(cd.get("best_budget_brier", "?")),
            str(cd.get("best_budget_nll", "?")),
            "YES" if cd.get("acc_brier_diverge") else "NO",
        ])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    tex = latex_table(
        ["Cell", "Dataset", r"$b^\star_{\text{acc}}$", r"$b^\star_{\text{Brier}}$",
         r"$b^\star_{\text{NLL}}$", r"$b^\star_{\text{acc}}\neq b^\star_{\text{Brier}}$"],
        rows,
        caption=r"Objective divergence: optimal reasoning budget differs by metric across model/dataset pairs.",
        label="tab:a05_objective",
    )
    (TAB_DIR / "a05_objective.tex").write_text(tex)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A08 — Global vs Item-Adaptive
# ---------------------------------------------------------------------------

def report_a08(d: dict) -> str:
    lines = ["\n## A08 — Global vs Item-Adaptive Policy\n"]
    headers = ["Cell", "Dataset", "Oracle saving", "Mean oracle B",
               "VISTA-Global Brier", "VISTA-Global saving", "Best-fixed B"]
    rows = []
    for cell_key, cd in d.items():
        if cell_key.startswith("_"): continue
        ds_name = cd.get("_meta", {}).get("ds_name", cell_key)
        g0 = cd.get("gate0", {})
        vg = cd.get("vista_global", {})
        rows.append([
            _latex_esc(cell_key), ds_name,
            _fmt(g0.get("saving_vs_max"), ".3f"),
            _fmt(g0.get("mean_oracle"), ".0f"),
            _fmt(vg.get("mean_brier"), ".4f"),
            _fmt(vg.get("token_saving"), ".3f"),
            str(cd.get("best_fixed_b", "?")),
        ])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    tex = latex_table(
        ["Cell", "Dataset",
         r"Oracle saving $\uparrow$", r"$\bar{b}_{\text{oracle}}$",
         r"VISTA-Global Brier", r"VISTA-Global saving $\uparrow$", r"Best-fixed $b$"],
        rows,
        caption=r"Gate-0 oracle token saving and VISTA-Global policy at matched quality.",
        label="tab:a08_global_adaptive",
    )
    (TAB_DIR / "a08_global_adaptive.tex").write_text(tex)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A09 — Matched-Quality Policy Comparison
# ---------------------------------------------------------------------------

def report_a09(d: dict) -> str:
    lines = ["\n## A09 — Matched-Quality Policy Comparison\n"]
    for cell_key, cd in d.items():
        if cell_key.startswith("_"): continue
        ds_name = cd.get("_meta", {}).get("ds_name", cell_key)
        policies = cd.get("policies", {})
        if not policies: continue
        lines.append(f"\n**{cell_key}** ({ds_name}), B_max Brier={_fmt(cd.get('bmax_brier'), '.4f')}\n")
        ph = ["Policy", "Brier", "Acc", "NLL", "Tokens", "Saving", "ΔBrier"]
        lines.append("| " + " | ".join(ph) + " |")
        lines.append("|" + "|".join(["---"] * len(ph)) + "|")
        for name, pm in policies.items():
            if not pm.get("feasible", True): continue
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                _fmt(pm.get("mean_brier"), ".4f"),
                _fmt(pm.get("accuracy"), ".3f"),
                _fmt(pm.get("mean_nll"), ".3f"),
                _fmt(pm.get("mean_tokens"), ".0f"),
                _fmt(pm.get("token_saving"), ".3f"),
                _fmt(pm.get("brier_delta_vs_bmax"), "+.4f"),
            ))

    # Generate LaTeX for m1_ds1 only
    if "m1_ds1" in d:
        cd = d["m1_ds1"]
        policies = cd.get("policies", {})
        key_policies = [
            "Fixed B=256", "Fixed B=512", "Fixed B=1024", "Fixed B=2048",
            "Fixed B=4096", "Fixed B=8192",
            "Conf≥0.90", "Conf≥0.85", "Conf≥0.80",
            "VISTA-Global(0.85,0.45)", "VISTA-Global(0.90,0.50)",
            "Oracle-Brier",
        ]
        rows = []
        for name in key_policies:
            pm = policies.get(name, {})
            if not pm.get("feasible", True): continue
            rows.append([
                name.replace("≥", r"$\geq$"),
                _fmt(pm.get("mean_brier"), ".4f"),
                _fmt(pm.get("accuracy"), ".3f"),
                _fmt(pm.get("mean_tokens"), ".0f"),
                _fmt(pm.get("token_saving"), ".3f"),
                _fmt(pm.get("brier_delta_vs_bmax"), "+.4f"),
            ])
        tex = latex_table(
            ["Policy", r"Brier $\downarrow$", r"Acc $\uparrow$",
             r"Tokens $\downarrow$", r"Saving $\uparrow$", r"$\Delta$Brier vs $b_{\max}$"],
            rows,
            caption=r"Matched-quality policy comparison on M1 (Qwen3-8B) / MMLU-Pro.",
            label="tab:a09_matched_quality",
        )
        (TAB_DIR / "a09_matched_quality.tex").write_text(tex)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A20 — Component Removal
# ---------------------------------------------------------------------------

def report_a20(d: dict) -> str:
    lines = ["\n## A20 — Component Removal\n"]
    for cell_key, cd in d.items():
        if cell_key.startswith("_"): continue
        ds_name = cd.get("_meta", {}).get("ds_name", cell_key)
        variants = cd.get("variants", {})
        oracle_brier = cd.get("oracle_mean_brier", float("nan"))
        lines.append(f"\n**{cell_key}** ({ds_name}), oracle Brier={_fmt(oracle_brier, '.4f')}\n")
        headers = ["Variant", "Brier", "Saving", "Regret vs Oracle", "Seq Regret"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for vname, vm in variants.items():
            seq = vm.get("sequential", {})
            lines.append("| {} | {} | {} | {} | {} |".format(
                vname,
                _fmt(vm.get("mean_brier"), ".4f"),
                _fmt(vm.get("token_saving"), ".3f"),
                _fmt(vm.get("mean_regret_vs_oracle"), ".4f"),
                _fmt(seq.get("mean_regret"), ".4f"),
            ))

    # LaTeX for m1_ds1
    if "m1_ds1" in d:
        cd = d["m1_ds1"]
        variants = cd.get("variants", {})
        rows = []
        order = [
            "VISTA_Full", "VISTA_minus_Audit", "VISTA_minus_IPW", "VISTA_minus_CS",
            "VISTA_minus_ItemAdapt", "VISTA_minus_Prequential", "VISTA_minus_Feedback",
            "Fixed_BestGlobal", "Fixed_Bmax",
        ]
        labels = {
            "VISTA_Full":              r"\textbf{VISTA (Full)}",
            "VISTA_minus_Audit":       r"$-$ Audit",
            "VISTA_minus_IPW":         r"$-$ IPW",
            "VISTA_minus_CS":          r"$-$ CS",
            "VISTA_minus_ItemAdapt":   r"$-$ Item-Adapt",
            "VISTA_minus_Prequential": r"$-$ Prequential",
            "VISTA_minus_Feedback":    r"$-$ Feedback",
            "Fixed_BestGlobal":        r"Fixed best-$b$",
            "Fixed_Bmax":              r"Fixed $b_{\max}$",
        }
        for vname in order:
            vm = variants.get(vname, {})
            if not vm: continue
            seq = vm.get("sequential", {})
            rows.append([
                labels.get(vname, vname),
                _fmt(vm.get("mean_brier"), ".4f"),
                _fmt(vm.get("token_saving"), ".3f"),
                _fmt(vm.get("mean_regret_vs_oracle"), ".4f"),
                _fmt(seq.get("mean_regret"), ".4f") if seq else "—",
            ])
        tex = latex_table(
            ["Variant", r"Brier $\downarrow$", r"Saving $\uparrow$",
             r"$\Delta$Brier(oracle)", r"Seq.\ regret"],
            rows,
            caption=r"Component-removal ablation: VISTA vs.\ each component removed (M1/MMLU-Pro).",
            label="tab:a20_component_removal",
        )
        (TAB_DIR / "a20_component_removal.tex").write_text(tex)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A06 — Prefix Signal heatmap table
# ---------------------------------------------------------------------------

def report_a06(d: dict) -> str:
    lines = ["\n## A06 — Prefix Signal Ablation\n"]
    SIGNALS = ["confidence", "entropy", "margin", "answer_stability", "conf_slope"]

    for cell_key, cd in d.items():
        if cell_key.startswith("_"): continue
        if "error" in cd: continue
        ds_name = cd.get("_meta", {}).get("ds_name", cell_key)
        edges = cd.get("edges", [])
        if not edges: continue
        lines.append(f"\n**{cell_key}** ({ds_name}) | gate1_pass={cd.get('gate1_pass')}\n")
        headers = ["Edge"] + [f"{s}_AUC" for s in SIGNALS[:4]]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for e in edges:
            bl, bh = e["b_lo"], e["b_hi"]
            row = [f"{bl}→{bh}"] + [_fmt(e.get(f"{s}_auc"), ".3f") for s in SIGNALS[:4]]
            lines.append("| " + " | ".join(str(c) for c in row) + " |")

    # LaTeX heatmap table (aggregated across cells)
    # Build mean AUC per (model, edge, signal)
    cell_groups = [
        ("m1", ["m1_ds1", "m1_ds2", "m1_ds3", "m1_ds4"]),
        ("m2", ["m2_ds1", "m2_ds2", "m2_ds3", "m2_ds4"]),
    ]
    from collections import defaultdict
    import statistics as stats

    all_edges = []
    for cd in d.values():
        if isinstance(cd, dict) and "edges" in cd:
            for e in cd["edges"]:
                pair = (e["b_lo"], e["b_hi"])
                if pair not in all_edges:
                    all_edges.append(pair)

    rows = []
    for group_name, cell_keys in cell_groups:
        for (bl, bh) in all_edges:
            row = [f"M{group_name[-1]} {bl}→{bh}"]
            for sig in SIGNALS[:4]:
                vals = []
                for ck in cell_keys:
                    cd = d.get(ck, {})
                    for e in cd.get("edges", []):
                        if e.get("b_lo") == bl and e.get("b_hi") == bh:
                            v = e.get(f"{sig}_auc", float("nan"))
                            if not math.isnan(v):
                                vals.append(v)
                row.append(_fmt(stats.mean(vals), ".3f") if vals else "—")
            rows.append(row)

    if rows:
        tex = latex_table(
            ["Edge", "Confidence AUC", "Entropy AUC", "Margin AUC", "Stability AUC"],
            rows,
            caption=r"Gate-1 prefix signal AUC (mean over datasets) for each budget edge. AUC$>0.60$ indicates useful stopping signal.",
            label="tab:a06_signal_heatmap",
        )
        (TAB_DIR / "a06_signal_heatmap.tex").write_text(tex)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A12 — Distribution Shift
# ---------------------------------------------------------------------------

def report_a12(d: dict) -> str:
    lines = ["\n## A12 — Distribution Shift\n"]
    for model_key, md in d.items():
        if model_key.startswith("_"): continue
        shift_results = md.get("shift_results", [])
        if not shift_results: continue
        lines.append(f"\n**{model_key}**\n")
        headers = ["Shift", "Fixed loss", "Preq loss", "Oracle loss", "Preq advantage"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in shift_results:
            if r.get("skipped"): continue
            lines.append("| {} | {} | {} | {} | {} |".format(
                r.get("label", "?"),
                _fmt(r.get("fixed_mean_loss"), ".4f"),
                _fmt(r.get("preq_mean_loss"), ".4f"),
                _fmt(r.get("oracle_mean_loss"), ".4f"),
                _fmt(r.get("preq_advantage_over_fixed"), "+.4f"),
            ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A13 — Cross-Model Transfer
# ---------------------------------------------------------------------------

def report_a13(d: dict) -> str:
    lines = ["\n## A13 — Cross-Model Transfer\n"]
    results = d.get("transfer_results", [])
    headers = ["Dataset", "M1 tau", "M1 tuned Brier", "M2 from M1 Brier",
               "M1→M2 degradation", "M2 tau", "M2 tuned Brier", "M2→M1 degradation"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in results:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("ds_tag", "?"),
            _fmt(r.get("m1_best_tau"), ".2f"),
            _fmt(r.get("m1_tuned", {}).get("brier"), ".4f"),
            _fmt(r.get("m2_transfer_from_m1", {}).get("brier"), ".4f"),
            _fmt(r.get("m1to_m2_brier_degradation"), "+.4f"),
            _fmt(r.get("m2_best_tau"), ".2f"),
            _fmt(r.get("m2_tuned", {}).get("brier"), ".4f"),
            _fmt(r.get("m2to_m1_brier_degradation"), "+.4f"),
        ))

    tex_rows = []
    for r in results:
        tex_rows.append([
            _latex_esc(r.get("ds_tag", "?")),
            _fmt(r.get("m1_best_tau"), ".2f"),
            _fmt(r.get("m1_tuned", {}).get("brier"), ".4f"),
            _fmt(r.get("m2_transfer_from_m1", {}).get("brier"), ".4f"),
            _fmt(r.get("m1to_m2_brier_degradation"), "+.4f"),
        ])
    if tex_rows:
        tex = latex_table(
            ["Dataset", r"$\tau_{M1}$", "M1 tuned", "M2 w/ $\\tau_{M1}$", r"$\Delta$Brier"],
            tex_rows,
            caption=r"Cross-model transfer: M1-calibrated threshold $\tau$ applied to M2 without re-tuning.",
            label="tab:a13_cross_model",
        )
        (TAB_DIR / "a13_cross_model.tex").write_text(tex)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A19 — Bootstrap CI summary
# ---------------------------------------------------------------------------

def report_a19(d: dict) -> str:
    lines = ["\n## A19 — Bootstrap Robustness\n"]
    headers = ["Cell", "Dataset", "Oracle saving (95% CI)", "Stream order CV"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for cell_key, cd in d.items():
        if cell_key.startswith("_"): continue
        ds_name = cd.get("_meta", {}).get("ds_name", cell_key)
        oci = cd.get("oracle_saving_ci", {})
        sos = cd.get("stream_order_sensitivity", {})
        pt   = _fmt(oci.get("point"), ".3f")
        lo   = _fmt(oci.get("lo"), ".3f")
        hi   = _fmt(oci.get("hi"), ".3f")
        cv   = _fmt(sos.get("cv"), ".4f")
        lines.append(f"| {cell_key} | {ds_name} | {pt} [{lo}, {hi}] | {cv} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claim verdict
# ---------------------------------------------------------------------------

def verdict_claims(all_results: dict) -> str:
    lines = ["\n## Claim Verdicts\n"]

    verdicts = {}

    # C1: Additional reasoning has model-, task-, budget-dependent marginal value
    d05 = all_results.get("a05")
    if d05:
        n_diverge = sum(1 for v in d05.values()
                        if isinstance(v, dict) and v.get("acc_brier_diverge"))
        verdicts["C1"] = ("SUPPORTED" if n_diverge > 0 else "PARTIALLY SUPPORTED",
                          f"{n_diverge}/8 cells show Acc≠Brier optimal budget")

    # C2: Accuracy-optimal and proper-score-optimal budgets can differ
    if d05:
        n_div = sum(1 for v in d05.values()
                    if isinstance(v, dict) and v.get("acc_brier_diverge"))
        verdicts["C2"] = ("SUPPORTED" if n_div > 0 else "NOT SUPPORTED",
                          f"{n_div}/8 cells where best-acc ≠ best-Brier budget")

    # C3: Large item-level allocation headroom
    d08 = all_results.get("a08")
    if d08:
        savings = [v.get("oracle_token_saving", 0) for v in d08.values()
                   if isinstance(v, dict) and "oracle_token_saving" in v]
        if savings:
            verdicts["C3"] = ("SUPPORTED" if min(savings) > 0.20 else "PARTIALLY SUPPORTED",
                              f"Oracle saving: min={min(savings):.3f} max={max(savings):.3f}")

    # C5: Randomized auditing necessary for counterfactual recovery
    d02 = all_results.get("a02")
    if d02:
        obs_biases = [v.get("observed_only", {}).get("mean_bias", 0)
                      for v in d02.values() if isinstance(v, dict)]
        ipw_biases = [abs(v.get("ipw", {}).get("mean_bias", 0))
                      for v in d02.values() if isinstance(v, dict)]
        if obs_biases and ipw_biases:
            verdicts["C5"] = ("SUPPORTED" if max(obs_biases) > 0.001 else "NOT SUPPORTED",
                              f"Obs-only max bias={max(obs_biases):.5f}, IPW max|bias|={max(ipw_biases):.6f}")

    # C6: IPW necessary for unbiased policy-risk estimation
    d01 = all_results.get("a01")
    if d01:
        naive_biases = []
        ipw_biases   = []
        for v in d01.values():
            if not isinstance(v, dict): continue
            for rrow in v.get("rho_sweep", []):
                if rrow.get("rho") == 0.20:
                    naive_biases.append(rrow.get("naive_mean_bias", 0))
                    ipw_biases.append(rrow.get("ipw_mean_bias", 0))
        if naive_biases:
            verdicts["C6"] = ("SUPPORTED" if max(naive_biases) > max(ipw_biases) * 2 else "PARTIALLY SUPPORTED",
                              f"Naive bias={sum(naive_biases)/len(naive_biases):.5f} "
                              f"vs IPW bias={sum(ipw_biases)/len(ipw_biases):.5f} at rho=0.20")

    # C7: CS reduces erroneous policy switching
    d03 = all_results.get("a03")
    if d03:
        cs_fprs     = [v.get("cs_false_positive_rate", 1) for v in d03.values() if isinstance(v, dict)]
        greedy_fprs = [v.get("greedy_false_positive_rate", 0) for v in d03.values() if isinstance(v, dict)]
        if cs_fprs and greedy_fprs:
            verdicts["C7"] = (
                "SUPPORTED" if sum(cs_fprs) < sum(greedy_fprs) else "NOT SUPPORTED",
                f"CS FPR mean={sum(cs_fprs)/len(cs_fprs):.4f} vs Greedy FPR mean={sum(greedy_fprs)/len(greedy_fprs):.4f}"
            )

    # C8: Prequential calibration adapts to distribution shift
    d12 = all_results.get("a12")
    if d12:
        advantages = []
        for model_key, md in d12.items():
            if not isinstance(md, dict): continue
            for row in md.get("shift_results", []):
                if not row.get("skipped"):
                    advantages.append(row.get("preq_advantage_over_fixed", 0.0))
        if advantages:
            n_pos = sum(1 for a in advantages if a > 0)
            mean_adv = sum(advantages) / len(advantages)
            if n_pos > len(advantages) // 2:
                verdict_c8 = "SUPPORTED"
            elif n_pos > 0:
                verdict_c8 = "PARTIALLY SUPPORTED"
            else:
                verdict_c8 = "NOT SUPPORTED"
            verdicts["C8"] = (verdict_c8,
                              f"Preq. advantage>0 in {n_pos}/{len(advantages)} shift scenarios, mean={mean_adv:+.4f}")

    for claim_id, (verdict, detail) in verdicts.items():
        lines.append(f"- **{claim_id}**: {verdict} — {detail}")

    not_computed = [c for c in ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]
                    if c not in verdicts]
    if not_computed:
        lines.append(f"\n_Claims not yet computable (pending results): {', '.join(not_computed)}_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(all_results: dict, no_figures: bool = False):
    if no_figures:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; skipping figures")
        return

    # Figure 1: A05 — Best budget by objective, per cell
    d05 = all_results.get("a05")
    if d05:
        cells = [(k, v) for k, v in d05.items() if isinstance(v, dict) and "budget_rows" in v]
        if cells:
            fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
            for ax, (cell_key, cd) in zip(axes.flatten(), cells):
                rows = cd["budget_rows"]
                budgets = [r["budget"] for r in rows]
                accs    = [r["accuracy"] for r in rows]
                briers  = [r["mean_brier"] for r in rows]
                nlls    = [r["mean_nll"] for r in rows]
                ax2 = ax.twinx()
                l1, = ax.plot(budgets, accs,   "b-o", ms=4, label="Acc")
                l2, = ax.plot(budgets, briers, "r-s", ms=4, label="Brier")
                l3, = ax2.plot(budgets, nlls,  "g--^", ms=4, label="NLL")
                ax.set_title(cell_key, fontsize=8)
                ax.set_xscale("log")
                ax.tick_params(axis="both", labelsize=7)
                ax2.tick_params(axis="both", labelsize=7)
            fig.suptitle("Objective divergence: Acc / Brier / NLL by budget", fontsize=10)
            fig.tight_layout()
            plt.savefig(FIG_DIR / "a05_objective_divergence.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  → figures/ablations/a05_objective_divergence.png")

    # Figure 2: A08 — Oracle saving per cell (bar chart)
    d08 = all_results.get("a08")
    if d08:
        cell_keys = [k for k in d08 if isinstance(d08[k], dict) and "oracle_token_saving" in d08[k]]
        savings = [d08[k]["oracle_token_saving"] for k in cell_keys]
        if cell_keys:
            fig, ax = plt.subplots(figsize=(9, 3.5))
            colors = ["steelblue" if k.startswith("m1") else "coral" for k in cell_keys]
            ax.bar(range(len(cell_keys)), savings, color=colors)
            ax.set_xticks(range(len(cell_keys)))
            ax.set_xticklabels(cell_keys, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Oracle token saving vs B_max", fontsize=9)
            ax.set_title("Gate-0: per-cell oracle compute saving opportunity", fontsize=10)
            ax.axhline(0.10, color="gray", linestyle="--", linewidth=0.8, label="10% threshold")
            ax.legend(fontsize=8)
            fig.tight_layout()
            plt.savefig(FIG_DIR / "a08_oracle_saving.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  → figures/ablations/a08_oracle_saving.png")

    # Figure 3: A01 — IPW vs Naive bias (left: m1_ds1 rho sweep; right: wrong-best rate all cells)
    d01 = all_results.get("a01")
    if d01:
        valid_cells = [k for k in d01 if isinstance(d01[k], dict) and d01[k].get("rho_sweep")]
        if valid_cells:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
            # Left panel: bias vs rho for m1_ds1 (representative)
            ref_cell = "m1_ds1" if "m1_ds1" in d01 else valid_cells[0]
            rho_rows = d01[ref_cell].get("rho_sweep", [])
            rhos       = [r["rho"] for r in rho_rows]
            ipw_bias   = [r.get("ipw_mean_bias", float("nan")) for r in rho_rows]
            naive_bias = [r.get("naive_mean_bias", float("nan")) for r in rho_rows]
            axes[0].plot(rhos, ipw_bias,   "b-o", label="IPW bias", ms=6)
            axes[0].plot(rhos, naive_bias, "r-s", label="Naive bias", ms=6)
            axes[0].set_xlabel(r"Audit rate $\rho$", fontsize=10)
            axes[0].set_ylabel("Mean |bias| in risk estimate", fontsize=9)
            axes[0].set_title(f"Bias vs $\\rho$ ({ref_cell})", fontsize=10)
            axes[0].legend(fontsize=9)
            # Right panel: wrong-best-budget rate at rho=0.20 across all cells
            cells_sorted = sorted(valid_cells)
            wrong_ipw   = []
            wrong_naive = []
            for ck in cells_sorted:
                rsweep = d01[ck].get("rho_sweep", [])
                r20 = next((r for r in rsweep if abs(r["rho"]-0.20)<0.01), None)
                wrong_ipw.append(r20.get("wrong_best_ipw_rate", float("nan")) if r20 else float("nan"))
                wrong_naive.append(r20.get("wrong_best_naive_rate", float("nan")) if r20 else float("nan"))
            x = range(len(cells_sorted))
            axes[1].bar([i - 0.18 for i in x], wrong_ipw,   0.35, label="IPW",   color="steelblue")
            axes[1].bar([i + 0.18 for i in x], wrong_naive, 0.35, label="Naive", color="tomato")
            axes[1].set_xticks(list(x))
            axes[1].set_xticklabels(cells_sorted, rotation=30, ha="right", fontsize=7)
            axes[1].set_ylabel(r"Wrong-best-budget rate ($\rho=0.20$)", fontsize=9)
            axes[1].set_title("Policy selection error rate, all cells", fontsize=10)
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(fontsize=9)
            fig.tight_layout()
            plt.savefig(FIG_DIR / "a01_ipw_vs_naive.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  → figures/ablations/a01_ipw_vs_naive.png")

    # Figure 4: A20 — Component removal brier bar chart (m1_ds1)
    d20 = all_results.get("a20")
    if d20 and "m1_ds1" in d20:
        variants = d20["m1_ds1"].get("variants", {})
        order = ["VISTA_Full", "VISTA_minus_Audit", "VISTA_minus_IPW", "VISTA_minus_CS",
                 "VISTA_minus_ItemAdapt", "VISTA_minus_Prequential", "Fixed_BestGlobal", "Fixed_Bmax"]
        names  = [v for v in order if v in variants]
        briers = [variants[v].get("mean_brier", float("nan")) for v in names]
        if any(not math.isnan(b) for b in briers):
            short_names = [n.replace("VISTA_", "VISTA-").replace("minus_", "-").replace("Fixed_", "Fixed ") for n in names]
            fig, ax = plt.subplots(figsize=(9, 3.5))
            colors = ["steelblue"] + ["coral"] * (len(names) - 3) + ["gray", "lightgray"]
            valid = [(n, b, c) for n, b, c in zip(short_names, briers, colors) if not math.isnan(b)]
            ax.bar(range(len(valid)), [b for _, b, _ in valid],
                   color=[c for _, _, c in valid])
            ax.set_xticks(range(len(valid)))
            ax.set_xticklabels([n for n, _, _ in valid], rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Mean Brier", fontsize=9)
            ax.set_title("A20: VISTA component removal (M1/MMLU-Pro)", fontsize=10)
            fig.tight_layout()
            plt.savefig(FIG_DIR / "a20_component_removal.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  → figures/ablations/a20_component_removal.png")

    # Figure 5: A06 — Signal AUC heatmap
    d06 = all_results.get("a06")
    if d06:
        SIGNALS = ["confidence", "entropy", "margin", "answer_stability", "conf_slope"]
        edge_labels = ["256→512", "512→1024", "1024→2048", "2048→4096", "4096→8192"]
        cell_keys = [k for k in d06 if isinstance(d06[k], dict) and "edges" in d06[k]]
        if cell_keys:
            # Build matrix: cell × edge × signal
            n_cells = len(cell_keys)
            n_edges = 5
            n_sigs  = len(SIGNALS)
            mat = np.full((n_cells, n_sigs, n_edges), np.nan)
            for ci, ck in enumerate(cell_keys):
                edges = d06[ck].get("edges", [])
                for ej, edge in enumerate(edges):
                    if ej >= n_edges: break
                    for si, sig in enumerate(SIGNALS):
                        mat[ci, si, ej] = edge.get(f"{sig}_auc", np.nan)
            # Mean across cells
            mean_mat = np.nanmean(mat, axis=0)  # (n_sigs, n_edges)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, group_label, idx_slice in zip(
                axes, ["M1 (Qwen3-8B)", "M2 (DeepSeek-R1-Llama)"],
                [slice(0, 4), slice(4, 8)]
            ):
                group_cells = cell_keys[idx_slice]
                if not group_cells:
                    ax.set_visible(False)
                    continue
                group_mat = np.nanmean(mat[idx_slice], axis=0)
                im = ax.imshow(group_mat, vmin=0.4, vmax=0.8, cmap="RdYlGn", aspect="auto")
                ax.set_xticks(range(n_edges))
                ax.set_xticklabels(edge_labels, rotation=30, fontsize=8)
                ax.set_yticks(range(n_sigs))
                ax.set_yticklabels(SIGNALS, fontsize=8)
                ax.set_title(f"Gate-1 Signal AUC — {group_label}", fontsize=9)
                for r in range(n_sigs):
                    for c in range(n_edges):
                        v = group_mat[r, c]
                        if not np.isnan(v):
                            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                    color="black" if 0.45 < v < 0.75 else "white")
                plt.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle("A06: Prefix signal AUC heatmap (mean over datasets)", fontsize=10)
            fig.tight_layout()
            plt.savefig(FIG_DIR / "a06_signal_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  → figures/ablations/a06_signal_heatmap.png")


# ---------------------------------------------------------------------------
# Master ablation status table
# ---------------------------------------------------------------------------

def make_master_table(all_results: dict) -> str:
    STATUS_MAP = {
        "01": ("A01 IPW vs Naive",                  "P1", "C5,C6"),
        "02": ("A02 Counterfactual Missingness",     "P0", "C5"),
        "03": ("A03 CS vs Greedy",                   "P0", "C7"),
        "04": ("A04 Audit Rate Sensitivity",          "P1", "C5,C6"),
        "05": ("A05 Objective Divergence",            "P0", "C1,C2"),
        "06": ("A06 Prefix Signal",                   "P0", "C4"),
        "07": ("A07 Signal Combination",              "P1", "C4"),
        "08": ("A08 Global vs Item-Adaptive",         "P0", "C3,C9"),
        "09": ("A09 Matched-Quality Policy",          "P0", "C9"),
        "10": ("A10 Lambda Sensitivity",              "P1", "C9,C10"),
        "11": ("A11 Prequential vs Fixed",            "P1", "C8"),
        "12": ("A12 Distribution Shift",             "P0", "C8"),
        "13": ("A13 Cross-Model Transfer",           "P0", "C4"),
        "14": ("A14 Cross-Dataset Transfer",         "P1", "C4"),
        "15": ("A15 Forced vs Independent (BLOCKED)","P2", "—"),
        "16": ("A16 Delayed Label Latency",          "P1", "C8"),
        "17": ("A17 Policy Family Size",             "P2", "C7"),
        "18": ("A18 Audit Schedule",                 "P1", "C5"),
        "19": ("A19 Bootstrap Robustness",           "P1", "all"),
        "20": ("A20 Component Removal",              "P0", "C5-C9"),
    }

    lines = ["\n## Master Ablation Status\n"]
    headers = ["ID", "Ablation", "Tier", "Claims", "Status", "Cells done"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for aid, (label, tier, claims) in STATUS_MAP.items():
        data = all_results.get(aid)
        if data is None:
            status = "PENDING"
            cells_done = "0/8"
        elif aid == "15":
            status = "BLOCKED (GPU)"
            cells_done = "N/A"
        elif aid in ("12", "13", "14"):
            status = "DONE" if data else "PENDING"
            cells_done = "all" if data else "0"
        else:
            n_cells = sum(1 for k, v in data.items()
                          if isinstance(v, dict) and not k.startswith("_"))
            status = "DONE" if n_cells == 8 else (f"PARTIAL {n_cells}/8" if n_cells > 0 else "PENDING")
            cells_done = f"{n_cells}/8"
        lines.append(f"| A{aid} | {label} | {tier} | {claims} | {status} | {cells_done} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    # Load all available results
    all_results = {}
    for aid in [f"{i:02d}" for i in range(1, 21)]:
        fname = f"a{aid}.json"
        d = load(fname)
        if d:
            all_results[aid] = d
            print(f"  Loaded {fname}")
        else:
            print(f"  Missing {fname}")

    report_sections = []
    report_sections.append("# VISTA Ablation Campaign — Analysis Report\n")
    report_sections.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n")

    # Status table always first
    report_sections.append(make_master_table(all_results))

    # Per-ablation sections (only if data available)
    handlers = {
        "01": report_a01, "02": report_a02, "03": report_a03,
        "05": report_a05, "06": report_a06, "08": report_a08,
        "09": report_a09, "12": report_a12, "13": report_a13,
        "19": report_a19, "20": report_a20,
    }
    for aid, fn in sorted(handlers.items()):
        if aid in all_results:
            try:
                report_sections.append(fn(all_results[aid]))
            except Exception as e:
                report_sections.append(f"\n## A{aid} — ERROR: {e}\n")

    # Claim verdicts
    report_sections.append(verdict_claims(all_results))

    # Write report
    full_report = "\n".join(report_sections)
    REPORT_MD.write_text(full_report)
    print(f"\nReport → {REPORT_MD.relative_to(ROOT)}")

    # Figures
    print("Generating figures...")
    make_figures(all_results, no_figures=args.no_figures)

    # LaTeX tables for missing ablations
    print(f"\nLaTeX tables written to {TAB_DIR.relative_to(ROOT)}/")
    for tex_file in sorted(TAB_DIR.glob("*.tex")):
        print(f"  {tex_file.name}")


if __name__ == "__main__":
    main()

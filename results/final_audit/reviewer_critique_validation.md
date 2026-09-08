# Reviewer Critique Validation Report
Generated: 2026-08-31 | Paper: VISTA — Value-Informed Sequential Test-time Allocation

---

## Criticism 1: Table 11 component ablation is meaningless (all values identical)

**Status: CONFIRMED**

A20 data for m1/ds1 (tau_conf=0.85, lambda=0, rho=0.2, n=1000):
- VISTA_Full:    brier=0.6831, tokens=1273, saving=0.845
- -Audit:        brier=0.6831, tokens=1273, saving=0.845  (IDENTICAL)
- -IPW:          brier=0.6831, tokens=1273, saving=0.845  (IDENTICAL)
- -CS:           brier=0.6831, tokens=1273, saving=0.845  (IDENTICAL)
- -ItemAdapt:    brier=0.4917, tokens=8192, saving=0.000  (different — the meaningful split)

**Root cause:** At lambda=0, the prequential monitor never triggers a policy switch.
Removing Audit/IPW/CS changes the *estimation quality* but not the *deployed policy outcome*,
because the monitor's decision is always "stay at current policy" regardless.
The component removal table cannot show monitoring effects when monitoring never fires.

**Fix:** Restructure Table 11 to separate:
(a) monitoring components — show via A01 (IPW bias reduction) and A03 (CS FPR/power)
(b) item-level adaptation (conf threshold) — A20 shows the meaningful split

---

## Criticism 2: Table 9 IPW wrong-best selection inconsistent

**Status: CONFIRMED**

At rho=0.2, wrong-best selection rates (200 permutations):

| Cell        | Naive_wrong | IPW_wrong | IPW better selection? |
|-------------|-------------|-----------|----------------------|
| M1/MMLU-Pro | 0.000       | 0.230     | No (naive wins)       |
| M1/ARC      | 1.000       | 0.630     | Yes (reduced but not 0)|
| M1/MedMCQA  | 1.000       | 0.555     | Yes (reduced but not 0)|
| M1/MedQA    | 1.000       | 0.455     | Yes (reduced but not 0)|
| M2/MMLU-Pro | 1.000       | 0.525     | Yes (reduced but not 0)|
| M2/ARC      | 0.000       | 0.600     | No (naive wins)       |
| M2/MedMCQA  | 1.000       | 0.185     | Yes (reduced but not 0)|
| M2/MedQA    | 0.000       | 0.525     | No (naive wins)       |

IPW consistently reduces bias in ALL cells (bias ratio 2.9x–33x).
BUT: IPW does NOT "eliminate" wrong-best — at rho=0.2, wrong-best rates remain 18.5%–63.0%.
3/8 cells: naive has lower (or equal) wrong-best selection rate than IPW.

**Current paper claim (line 1198):** "eliminates wrong-best on affected cells" — FALSE.

**Correct claim:** "IPW consistently reduces estimation bias; naive rank-preserving on 3/8 cells (naturally identified best budget); IPW wrong-best rate reduced from 100% to 18.5%–63.0% on affected cells."

**Note:** IPW adds variance at finite rho, which can INCREASE wrong-best rate in cells where
the naive estimator happens to select correctly (m1_ds1: best_fixed=8192, naive always sees
b=8192 natural continuations as best, so wrong-best=0; IPW noise > bias correction benefit here).

---

## Criticism 3: CS FPR=0 achieved by never deciding

**Status: CONFIRMED**

CS mean time-to-decide vs stream length (n_items=1000, n_perm=200):

| Cell        | CS FPR | Greedy FPR | CS TTD | Stream N | CS power |
|-------------|--------|------------|--------|----------|----------|
| M1/MMLU-Pro | 0.000  | 0.070      | 1000   | 1000     | ~0       |
| M1/ARC      | 0.000  | 0.480      | 979    | 1000     | ~0       |
| M1/MedMCQA  | 0.000  | 0.400      | 500    | 1000     | ~0       |
| M1/MedQA    | 0.000  | 0.140      | 500    | 1000     | ~0       |
| M2/MMLU-Pro | 0.000  | 0.585      | 990    | 1000     | ~0       |
| M2/ARC      | 0.000  | 0.255      | 990    | 1000     | ~0       |
| M2/MedMCQA  | 0.000  | 0.000      | 459    | 1000     | partial  |
| M2/MedQA    | 0.000  | 0.000      | 460    | 1000     | partial  |

CS FPR=0.000 is mathematically guaranteed by the anytime-valid construction —
it is NOT evidence that CS is making correct decisions. CS achieves FPR=0 by
never triggering a switch decision within the available stream (n=200 items at rho=0.2 = ~40 audits).

**Current paper:** Already acknowledges this ("time-to-decide equal to the full stream length").
**Fix needed:** Be more explicit that CS power≈0 at this deployment horizon.
Remove framing that suggests FPR=0 is a positive performance result in isolation.

---

## Criticism 4: Oracle C is too loose; gap with Oracle B large

**Status: PARTIALLY CONFIRMED**

Oracle comparison (m1/ds1 = M1/MMLU-Pro as example):
- Oracle B (per-item quality match, gate0.json): mean_budget=2431, saving=70%
- Oracle C (population constraint, stream_results.json): mean_budget=461, saving=94%
- Gap: 2431 vs 461 = Oracle C is 5.3x more optimistic than Oracle B

Paper already shows both Oracle B and C in the text and tab:oracle.
The headline abstract uses Oracle A (30.1% Brier reduction) and Oracle C (74-97% savings).
Oracle B savings (70% for m1/ds1) are more conservative and per-item defensible.

**Remaining issue:** Oracle C savings (74-97%) headline the abstract without explicit
"population-level constraint" label. Reader may assume these are per-item guarantees.
**Fix:** Add "population-level" qualifier to all Oracle C references in abstract/conclusion.

---

## Criticism 5: Related Work has no citations

**Status: CONFIRMED**

mvtcs.bib is empty (1 byte, newline only). No \cite commands exist in paper.
Related Work section (lines 286-325) has no citations.

**Fix:** Build real bibliography with ~30 verified references.

---

## Criticism 6: A6/A7/A8 evidence incomplete

**Status: PARTIALLY CONFIRMED**

- A6 (prefix signals): Data exists in a06.json, results present in Section 7.4 with AUC numbers. OK.
- A7 (audit-rate sensitivity): rho_sweep.json has full data but no dedicated paper table.
  Current paper has Section 7.8 text description but no table with CIs.
  Fix: Add rho sensitivity table with net_saving + CIs at rho ∈ {0.05,0.10,0.20,0.40}.
- A8 (forced checkpoint validity): a08.json exists, Section 7.10 discusses it. OK.

---

## Criticism 7: Statistical uncertainty missing

**Status: CONFIRMED**

Tab:main shows point estimates only. Lambda sweep has 95% CIs (n=200 permutations):
- m1/ds1 at lambda=0.01: net_saving=0.0319 [0.0267, 0.0371] → CI exists, not shown
- m1/ds2 at lambda=0.01: net_saving=0.3448 [0.3214, 0.3681]
- m1/ds4 at lambda=0.01: net_saving=0.0699 [0.0597, 0.0802]

**Fix:** Add CI column to Tab:main for VISTA rows.

---

## Criticism 8: Proposition 1 overclaimed as major theorem

**Status: NOT CONFIRMED**

Paper frames Proposition 1 as "non-identifiability result motivating target shift."
Language is appropriate: "formalises this" — not "proves a deep theorem."
Contribution list (line 266+) does not headline Proposition 1 as main novelty.
This criticism does not apply to the current manuscript.

---

## Summary

| Criticism | Status | Action Required |
|-----------|--------|-----------------|
| C1: Table 11 identical values | CONFIRMED | Restructure: separate item-level vs monitoring ablation |
| C2: IPW "eliminates" wrong-best | CONFIRMED | Fix text: "reduces" not "eliminates"; 3/8 naive wins |
| C3: CS FPR=0 via never deciding | CONFIRMED | Add power=0 framing explicitly |
| C4: Oracle B vs C gap | PARTIALLY | Add "population-level" qualifier to Oracle C |
| C5: No citations | CONFIRMED | Build bibliography |
| C6: A7 audit-rate table missing | PARTIALLY | Add rho sensitivity table with CIs |
| C7: No CIs in main tables | CONFIRMED | Add CIs to Tab:main VISTA rows |
| C8: Proposition overclaimed | NOT CONFIRMED | No action needed |

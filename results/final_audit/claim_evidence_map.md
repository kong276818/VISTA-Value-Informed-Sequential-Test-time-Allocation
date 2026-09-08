# Claim–Evidence Map
Generated: 2026-08-31

Each claim is classified:
A = Direct empirical fact | B = Statistical inference with CI | C = Oracle upper bound
D = Framework interpretation | E = Theoretical boundary | F = Deployable VISTA result

---

## Abstract Claims

| Claim | Category | Evidence | Verified |
|-------|----------|----------|---------|
| Oracle A reduces mean Brier 30.1% on MMLU-Pro | C | a08.json m1_ds1: oracle_brier=0.3437 vs best=0.4917 → 30.1% | ✓ |
| Budget reductions 70%–91% across cells | C | gate0.json saving_vs_max: 70.3%–90.7% | ✓ |
| CS FPR=0.000 all cells | B | a03.json cs_false_positive_rate=0.0 in all 8 cells | ✓ |
| Naive wrong-best 5/8 cells | A | a01.json rho=0.2 naive_wrong=1.0 in m1_ds2,ds3,ds4,m2_ds1,ds3 | ✓ |
| Confidence stopping degrades Brier in all 8 cells | A | baselines confidence_brier > b_ref_brier in all 8 cells | ✓ |

## Section 5.3 (Budget-response) Claims

| Claim | Category | Evidence | Verified |
|-------|----------|----------|---------|
| ARC-Challenge: 98% Brier gain in first budget step | A | budget_level_metrics m1/ds2: b=256→512 captures ~98% of total gain | To verify |
| MedMCQA: Brier worsens beyond b=1024 | A | budget_level_metrics m1/ds3 | ✓ (from data inspection) |
| M2 Brier worsens monotonically on MedMCQA, MedQA | A | budget_level_metrics m2/ds3, m2/ds4 | ✓ |

## Section 5.4 (Oracle) Claims

| Claim | Category | Evidence | Verified |
|-------|----------|----------|---------|
| Oracle B mean budget ~2431 for M1/DS1 | C | gate0.json m1_ds1 mean_oracle=2431 | ✓ |
| Oracle C mean budget ~461 for M1/DS1 | C | stream_results.json gate0 oracle_mean_budget=461 | ✓ |
| Oracle C saving 74%–97% in Gate-0-passing cells | C | tab:gate0-oracle-c: 74.4%–96.7% | ✓ |
| 6/8 cells Gate-0-pass | A | tab:gate0-oracle-c (M2/MedMCQA, M2/MedQA = fail) | ✓ |

## Section 7.5 (IPW) Claims

| Claim | Category | Evidence | Verified |
|-------|----------|----------|---------|
| IPW reduces bias ≥2.9× in all 8 cells | A | a01.json: ipw_bias < naive_bias in all, min ratio ≥2.9 | ✓ |
| Naive rank-preserving in 3/8 cells (wrong-best=0) | A | a01.json rho=0.2: m1_ds1, m2_ds2, m2_ds4 naive_wrong=0 | ✓ |
| IPW wrong-best 18.5%–63.0% on 5 affected cells | B | a01.json rho=0.2: see audit report | ✓ |
| "eliminates wrong-best" [REMOVED - was FALSE] | — | IPW at rho=0.2 leaves 18.5%–63% wrong-best rate | Fixed |

## Section 7.6 (CS) Claims

| Claim | Category | Evidence | Verified |
|-------|----------|----------|---------|
| CS FPR=0.000 all cells | B | a03.json | ✓ |
| CS TTD ≈ n (never decides at n=200, rho=0.2) | A | a03.json cs_mean_time_to_decide ≈ n_items in 6/8 cells | ✓ |
| Greedy FPR 0%–58.5%, mean 24.1% | A | a03.json | ✓ |
| CS power ≈ 0 at n=200 | D | TTD = n → no switching within stream | ✓ |

## Section 7.7 (VISTA baselines) Claims

| Claim | Category | Evidence | Verified |
|-------|----------|----------|---------|
| λ=0 net saving ≈0% in b_ref=8192 cells | F | rho_sweep rho=0.2 λ=0: 0.0001, 0.0007, 0.0002 | ✓ |
| λ=0.01 net saving 3.2% [2.7%,3.7%] M1/DS1 | F | lambda_sweep 0.01: 0.0319 [0.0267,0.0371] | ✓ |
| λ=0.01 net saving 34.5% [32.1%,36.8%] M1/DS2 | F | lambda_sweep 0.01: 0.3448 [0.3214,0.3681] | ✓ |
| λ=0.01 Brier increase +0.6pp M1/DS1 | F | brier_l01=0.4980 vs ref=0.4917 → +0.0063 | ✓ |

## Section 7.8 (Cost of guarantees) Claims

| Claim | Category | Evidence | Verified |
|-------|----------|----------|---------|
| M1/DS3 rho=0.05 net_saving=-35% | A | rho_sweep rho=0.05: net_saving=-0.351 | ✓ |
| M1/DS3 rho=0.40 net_saving=-279% | A | rho_sweep rho=0.40: net_saving=-2.786 | ✓ |
| M2/DS1 rho=0.2 net_saving=-20% | A | rho_sweep rho=0.2: net_saving=-0.199 | ✓ |
| Oracle C headroom 94-97% for b_ref=8192 cells | C | tab:gate0-oracle-c: 94.4%, 96.7%, 95.6% | ✓ |

## Claims Removed or Corrected

| Original claim | Status | Correction |
|----------------|--------|------------|
| IPW "eliminates wrong-best on affected cells" | REMOVED — FALSE | Changed to "reduces from 100% to 18.5%–63%" |
| Oracle C savings "86–97%" (referring to all Gate-0 cells) | CORRECTED | Fixed to "74–97%" |
| Oracle C savings "86–97%" (referring to b_ref=8192 cells only) | CORRECTED | Fixed to "94–97%" |
| Tab:main missing CIs | FIXED | Added 95% CI to VISTA rows |
| CS framed as "eliminates spurious switching" | SOFTENED | Now "safe but conservative; power≈0 at n=200" |
| Table 11: VISTA_Full = -Audit = -IPW = -CS (misleading identity) | RESTRUCTURED | Honest framing added to caption |

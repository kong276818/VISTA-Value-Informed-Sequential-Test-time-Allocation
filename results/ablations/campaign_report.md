# VISTA Ablation Campaign — Analysis Report

Generated: 2026-08-30T09:19:39.193641


## Master Ablation Status

| ID | Ablation | Tier | Claims | Status | Cells done |
|---|---|---|---|---|---|
| A01 | A01 IPW vs Naive | P1 | C5,C6 | DONE | 8/8 |
| A02 | A02 Counterfactual Missingness | P0 | C5 | DONE | 8/8 |
| A03 | A03 CS vs Greedy | P0 | C7 | DONE | 8/8 |
| A04 | A04 Audit Rate Sensitivity | P1 | C5,C6 | DONE | 8/8 |
| A05 | A05 Objective Divergence | P0 | C1,C2 | DONE | 8/8 |
| A06 | A06 Prefix Signal | P0 | C4 | DONE | 8/8 |
| A07 | A07 Signal Combination | P1 | C4 | DONE | 8/8 |
| A08 | A08 Global vs Item-Adaptive | P0 | C3,C9 | DONE | 8/8 |
| A09 | A09 Matched-Quality Policy | P0 | C9 | DONE | 8/8 |
| A10 | A10 Lambda Sensitivity | P1 | C9,C10 | DONE | 8/8 |
| A11 | A11 Prequential vs Fixed | P1 | C8 | DONE | 8/8 |
| A12 | A12 Distribution Shift | P0 | C8 | DONE | all |
| A13 | A13 Cross-Model Transfer | P0 | C4 | DONE | all |
| A14 | A14 Cross-Dataset Transfer | P1 | C4 | DONE | all |
| A15 | A15 Forced vs Independent (BLOCKED) | P2 | — | BLOCKED (GPU) | N/A |
| A16 | A16 Delayed Label Latency | P1 | C8 | DONE | 8/8 |
| A17 | A17 Policy Family Size | P2 | C7 | PARTIAL 2/8 | 2/8 |
| A18 | A18 Audit Schedule | P1 | C5 | DONE | 8/8 |
| A19 | A19 Bootstrap Robustness | P1 | all | DONE | 8/8 |
| A20 | A20 Component Removal | P0 | C5-C9 | DONE | 8/8 |

## A01 — IPW vs Naive


**m1_ds1** (MMLU-Pro, frac_natural_to_ref_b=0.122)
| rho | IPW bias | Naive bias | Wrong-best IPW | Wrong-best Naive |
|---|---|---|---|---|
| 0.05 | 0.05109 | 0.07079 | 0.465 | 0.000 |
| 0.1 | 0.03638 | 0.07079 | 0.360 | 0.000 |
| 0.2 | 0.02447 | 0.07079 | 0.230 | 0.000 |
| 0.4 | 0.01424 | 0.07079 | 0.065 | 0.000 |
| 1.0 | 0.00000 | 0.07079 | 0.000 | 0.000 |

**m1_ds2** (ARC-Challenge, frac_natural_to_ref_b=0.001)
| rho | IPW bias | Naive bias | Wrong-best IPW | Wrong-best Naive |
|---|---|---|---|---|
| 0.05 | 0.02224 | 0.23535 | 0.715 | 1.000 |
| 0.1 | 0.01629 | 0.23535 | 0.720 | 1.000 |
| 0.2 | 0.01031 | 0.23535 | 0.630 | 1.000 |
| 0.4 | 0.00642 | 0.23535 | 0.540 | 1.000 |
| 1.0 | 0.00000 | 0.23535 | 0.000 | 1.000 |

**m1_ds3** (MedMCQA, frac_natural_to_ref_b=0.018)
| rho | IPW bias | Naive bias | Wrong-best IPW | Wrong-best Naive |
|---|---|---|---|---|
| 0.05 | 0.05689 | 0.34441 | 0.710 | 1.000 |
| 0.1 | 0.03897 | 0.34441 | 0.675 | 1.000 |
| 0.2 | 0.02634 | 0.34441 | 0.555 | 1.000 |
| 0.4 | 0.01668 | 0.34441 | 0.380 | 1.000 |
| 1.0 | 0.00000 | 0.34441 | 0.000 | 1.000 |

**m1_ds4** (MedQA-USMLE, frac_natural_to_ref_b=0.028)
| rho | IPW bias | Naive bias | Wrong-best IPW | Wrong-best Naive |
|---|---|---|---|---|
| 0.05 | 0.06429 | 0.28715 | 0.680 | 1.000 |
| 0.1 | 0.04677 | 0.28715 | 0.565 | 1.000 |
| 0.2 | 0.03207 | 0.28715 | 0.455 | 1.000 |
| 0.4 | 0.01830 | 0.28715 | 0.460 | 1.000 |
| 1.0 | 0.00000 | 0.28715 | 0.000 | 1.000 |

**m2_ds1** (MMLU-Pro, frac_natural_to_ref_b=0.180)
| rho | IPW bias | Naive bias | Wrong-best IPW | Wrong-best Naive |
|---|---|---|---|---|
| 0.05 | 0.04587 | 0.04562 | 0.735 | 1.000 |
| 0.1 | 0.03324 | 0.04562 | 0.700 | 1.000 |
| 0.2 | 0.02092 | 0.04562 | 0.525 | 1.000 |
| 0.4 | 0.01267 | 0.04562 | 0.340 | 1.000 |
| 1.0 | 0.00000 | 0.04562 | 0.000 | 1.000 |

**m2_ds2** (ARC-Challenge, frac_natural_to_ref_b=0.007)
| rho | IPW bias | Naive bias | Wrong-best IPW | Wrong-best Naive |
|---|---|---|---|---|
| 0.05 | 0.02449 | 0.13682 | 0.770 | 0.000 |
| 0.1 | 0.01688 | 0.13682 | 0.780 | 0.000 |
| 0.2 | 0.01217 | 0.13682 | 0.600 | 0.000 |
| 0.4 | 0.00719 | 0.13682 | 0.510 | 0.000 |
| 1.0 | 0.00000 | 0.13682 | 0.000 | 0.000 |

**m2_ds3** (MedMCQA, frac_natural_to_ref_b=0.022)
| rho | IPW bias | Naive bias | Wrong-best IPW | Wrong-best Naive |
|---|---|---|---|---|
| 0.05 | 0.04224 | 0.23482 | 0.380 | 1.000 |
| 0.1 | 0.03223 | 0.23482 | 0.320 | 1.000 |
| 0.2 | 0.02035 | 0.23482 | 0.185 | 1.000 |
| 0.4 | 0.01210 | 0.23482 | 0.075 | 1.000 |
| 1.0 | 0.00000 | 0.23482 | 0.000 | 1.000 |

**m2_ds4** (MedQA-USMLE, frac_natural_to_ref_b=0.052)
| rho | IPW bias | Naive bias | Wrong-best IPW | Wrong-best Naive |
|---|---|---|---|---|
| 0.05 | 0.06311 | 0.10917 | 0.590 | 0.000 |
| 0.1 | 0.04346 | 0.10917 | 0.580 | 0.000 |
| 0.2 | 0.02986 | 0.10917 | 0.525 | 0.000 |
| 0.4 | 0.01749 | 0.10917 | 0.365 | 0.000 |
| 1.0 | 0.00000 | 0.10917 | 0.000 | 0.000 |

## A02 — Counterfactual Missingness

| Cell | Dataset | FullInfo Brier | ObsOnly bias | NaiveAudit bias | IPW bias |
|---|---|---|---|---|---|
| m1\_ds1 | MMLU-Pro | 0.4917 | 0.19135 | -0.00169 | -0.005329 |
| m1\_ds2 | ARC-Challenge | 0.0837 | 0.04688 | 0.00194 | 0.002762 |
| m1\_ds3 | MedMCQA | 0.6624 | 0.05291 | -0.00396 | -0.004034 |
| m1\_ds4 | MedQA-USMLE | 0.4285 | 0.21754 | -0.01093 | -0.010747 |
| m2\_ds1 | MMLU-Pro | 0.8875 | 0.07286 | -0.00289 | -0.009166 |
| m2\_ds2 | ARC-Challenge | 0.2338 | 0.07457 | -0.00321 | -0.005023 |
| m2\_ds3 | MedMCQA | 0.9292 | 0.01453 | -0.01078 | -0.010495 |
| m2\_ds4 | MedQA-USMLE | 0.8273 | -0.01486 | -0.01343 | -0.013392 |

## A03 — CS vs Greedy

| Cell | Dataset | CS FPR | Greedy FPR | Oracle best B |
|---|---|---|---|---|
| m1\_ds1 | MMLU-Pro | 0.0000 | 0.0700 | 8192 |
| m1\_ds2 | ARC-Challenge | 0.0000 | 0.4800 | 8192 |
| m1\_ds3 | MedMCQA | 0.0000 | 0.4000 | 1024 |
| m1\_ds4 | MedQA-USMLE | 0.0000 | 0.1400 | 8192 |
| m2\_ds1 | MMLU-Pro | 0.0000 | 0.5850 | 4096 |
| m2\_ds2 | ARC-Challenge | 0.0000 | 0.2550 | 2048 |
| m2\_ds3 | MedMCQA | 0.0000 | 0.0000 | 256 |
| m2\_ds4 | MedQA-USMLE | 0.0000 | 0.0000 | 256 |

## A05 — Objective Divergence (Acc vs Brier vs NLL)

| Cell | Dataset | Best-Acc B | Best-Brier B | Best-NLL B | Acc≠Brier? |
|---|---|---|---|---|---|
| m1\_ds1 | MMLU-Pro | 8192 | 8192 | 2048 | NO |
| m1\_ds2 | ARC-Challenge | 8192 | 8192 | 512 | NO |
| m1\_ds3 | MedMCQA | 4096 | 1024 | 256 | YES |
| m1\_ds4 | MedQA-USMLE | 8192 | 8192 | 2048 | NO |
| m2\_ds1 | MMLU-Pro | 4096 | 4096 | 256 | NO |
| m2\_ds2 | ARC-Challenge | 2048 | 2048 | 512 | NO |
| m2\_ds3 | MedMCQA | 4096 | 256 | 256 | YES |
| m2\_ds4 | MedQA-USMLE | 2048 | 256 | 256 | YES |

## A06 — Prefix Signal Ablation


**m1_ds1** (MMLU-Pro) | gate1_pass=True

| Edge | confidence_AUC | entropy_AUC | margin_AUC | answer_stability_AUC |
|---|---|---|---|---|
| 256→512 | 0.650 | 0.656 | 0.649 | — |
| 512→1024 | 0.720 | 0.718 | 0.722 | 0.512 |
| 1024→2048 | 0.788 | 0.790 | 0.787 | 0.560 |
| 2048→4096 | 0.813 | 0.814 | 0.814 | 0.569 |
| 4096→8192 | 0.813 | 0.814 | 0.813 | 0.538 |

**m1_ds2** (ARC-Challenge) | gate1_pass=True

| Edge | confidence_AUC | entropy_AUC | margin_AUC | answer_stability_AUC |
|---|---|---|---|---|
| 256→512 | 0.961 | 0.961 | 0.961 | — |
| 512→1024 | 0.968 | 0.968 | 0.968 | 0.621 |
| 1024→2048 | 0.951 | 0.951 | 0.952 | 0.575 |
| 2048→4096 | 0.926 | 0.925 | 0.929 | 0.596 |
| 4096→8192 | 0.805 | 0.804 | 0.807 | 0.571 |

**m1_ds3** (MedMCQA) | gate1_pass=True

| Edge | confidence_AUC | entropy_AUC | margin_AUC | answer_stability_AUC |
|---|---|---|---|---|
| 256→512 | 0.725 | 0.725 | 0.726 | — |
| 512→1024 | 0.794 | 0.793 | 0.795 | 0.550 |
| 1024→2048 | 0.699 | 0.699 | 0.700 | 0.564 |
| 2048→4096 | 0.759 | 0.759 | 0.758 | 0.498 |
| 4096→8192 | 0.640 | 0.640 | 0.638 | 0.529 |

**m1_ds4** (MedQA-USMLE) | gate1_pass=True

| Edge | confidence_AUC | entropy_AUC | margin_AUC | answer_stability_AUC |
|---|---|---|---|---|
| 256→512 | 0.783 | 0.784 | 0.783 | — |
| 512→1024 | 0.811 | 0.809 | 0.811 | 0.540 |
| 1024→2048 | 0.844 | 0.844 | 0.845 | 0.586 |
| 2048→4096 | 0.857 | 0.855 | 0.858 | 0.583 |
| 4096→8192 | 0.815 | 0.815 | 0.816 | 0.583 |

**m2_ds1** (MMLU-Pro) | gate1_pass=True

| Edge | confidence_AUC | entropy_AUC | margin_AUC | answer_stability_AUC |
|---|---|---|---|---|
| 256→512 | 0.455 | 0.461 | 0.455 | — |
| 512→1024 | 0.529 | 0.530 | 0.533 | 0.489 |
| 1024→2048 | 0.564 | 0.565 | 0.564 | 0.530 |
| 2048→4096 | 0.541 | 0.542 | 0.540 | 0.497 |
| 4096→8192 | 0.554 | 0.554 | 0.554 | 0.499 |

**m2_ds2** (ARC-Challenge) | gate1_pass=True

| Edge | confidence_AUC | entropy_AUC | margin_AUC | answer_stability_AUC |
|---|---|---|---|---|
| 256→512 | 0.768 | 0.769 | 0.767 | — |
| 512→1024 | 0.871 | 0.871 | 0.872 | 0.581 |
| 1024→2048 | 0.817 | 0.817 | 0.818 | 0.572 |
| 2048→4096 | 0.737 | 0.736 | 0.740 | 0.526 |
| 4096→8192 | 0.774 | 0.773 | 0.775 | 0.510 |

**m2_ds3** (MedMCQA) | gate1_pass=True

| Edge | confidence_AUC | entropy_AUC | margin_AUC | answer_stability_AUC |
|---|---|---|---|---|
| 256→512 | 0.487 | 0.490 | 0.487 | — |
| 512→1024 | 0.605 | 0.600 | 0.608 | 0.523 |
| 1024→2048 | 0.571 | 0.571 | 0.570 | 0.519 |
| 2048→4096 | 0.578 | 0.579 | 0.577 | 0.497 |
| 4096→8192 | 0.581 | 0.581 | 0.580 | 0.494 |

**m2_ds4** (MedQA-USMLE) | gate1_pass=True

| Edge | confidence_AUC | entropy_AUC | margin_AUC | answer_stability_AUC |
|---|---|---|---|---|
| 256→512 | 0.442 | 0.441 | 0.446 | — |
| 512→1024 | 0.542 | 0.547 | 0.539 | 0.526 |
| 1024→2048 | 0.625 | 0.624 | 0.626 | 0.512 |
| 2048→4096 | 0.589 | 0.590 | 0.590 | 0.524 |
| 4096→8192 | 0.598 | 0.600 | 0.596 | 0.505 |

## A08 — Global vs Item-Adaptive Policy

| Cell | Dataset | Oracle saving | Mean oracle B | VISTA-Global Brier | VISTA-Global saving | Best-fixed B |
|---|---|---|---|---|---|---|
| m1\_ds1 | MMLU-Pro | 0.643 | 2921 | 0.6831 | 0.845 | 8192 |
| m1\_ds2 | ARC-Challenge | 0.816 | 1506 | 0.1306 | 0.965 | 8192 |
| m1\_ds3 | MedMCQA | 0.664 | 2756 | 0.7153 | 0.937 | 1024 |
| m1\_ds4 | MedQA-USMLE | 0.571 | 3511 | 0.6461 | 0.940 | 8192 |
| m2\_ds1 | MMLU-Pro | 0.763 | 1943 | 0.9604 | 0.767 | 4096 |
| m2\_ds2 | ARC-Challenge | 0.847 | 1256 | 0.3083 | 0.953 | 2048 |
| m2\_ds3 | MedMCQA | 0.855 | 1185 | 0.9437 | 0.921 | 256 |
| m2\_ds4 | MedQA-USMLE | 0.809 | 1563 | 0.8124 | 0.872 | 256 |

## A09 — Matched-Quality Policy Comparison


**m1_ds1** (MMLU-Pro), B_max Brier=0.4917

| Policy | Brier | Acc | NLL | Tokens | Saving | ΔBrier |
|---|---|---|---|---|---|---|
| Fixed B=256 | 0.8324 | 0.477 | 3.186 | 256 | 0.969 | +0.3407 |
| Fixed B=512 | 0.7618 | 0.526 | 2.885 | 512 | 0.938 | +0.2701 |
| Fixed B=1024 | 0.6366 | 0.612 | 2.609 | 1024 | 0.875 | +0.1448 |
| Fixed B=2048 | 0.5282 | 0.696 | 2.458 | 2048 | 0.750 | +0.0365 |
| Fixed B=4096 | 0.5084 | 0.718 | 2.686 | 4096 | 0.500 | +0.0167 |
| Fixed B=8192 | 0.4917 | 0.738 | 2.763 | 8192 | 0.000 | +0.0000 |
| Conf≥0.90 | 0.6645 | 0.645 | 2.952 | 1453 | 0.823 | +0.1727 |
| Conf≥0.85 | 0.6831 | 0.631 | 2.929 | 1273 | 0.845 | +0.1914 |
| Conf≥0.80 | 0.7089 | 0.610 | 2.939 | 1136 | 0.861 | +0.2172 |
| VISTA-Global(0.85,0.45) | 0.6831 | 0.631 | 2.929 | 1273 | 0.845 | +0.1914 |
| VISTA-Global(0.90,0.50) | 0.6645 | 0.645 | 2.952 | 1453 | 0.823 | +0.1727 |
| Oracle-Brier | 0.3437 | 0.786 | 1.465 | 2921 | 0.643 | -0.1480 |

**m1_ds2** (ARC-Challenge), B_max Brier=0.0837

| Policy | Brier | Acc | NLL | Tokens | Saving | ΔBrier |
|---|---|---|---|---|---|---|
| Fixed B=256 | 0.1386 | 0.920 | 0.466 | 256 | 0.969 | +0.0548 |
| Fixed B=512 | 0.0978 | 0.946 | 0.355 | 512 | 0.938 | +0.0140 |
| Fixed B=1024 | 0.0885 | 0.951 | 0.371 | 1024 | 0.875 | +0.0047 |
| Fixed B=2048 | 0.0881 | 0.955 | 0.414 | 2048 | 0.750 | +0.0043 |
| Fixed B=4096 | 0.0860 | 0.956 | 0.462 | 4096 | 0.500 | +0.0022 |
| Fixed B=8192 | 0.0837 | 0.958 | 0.457 | 8192 | 0.000 | +0.0000 |
| Conf≥0.90 | 0.1232 | 0.937 | 0.457 | 289 | 0.965 | +0.0395 |
| Conf≥0.85 | 0.1306 | 0.932 | 0.464 | 283 | 0.965 | +0.0469 |
| Conf≥0.80 | 0.1314 | 0.931 | 0.455 | 275 | 0.966 | +0.0476 |
| VISTA-Global(0.85,0.45) | 0.1306 | 0.932 | 0.464 | 283 | 0.965 | +0.0469 |
| VISTA-Global(0.90,0.50) | 0.1232 | 0.937 | 0.457 | 289 | 0.965 | +0.0395 |
| Oracle-Brier | 0.0508 | 0.972 | 0.198 | 1506 | 0.816 | -0.0329 |

**m1_ds3** (MedMCQA), B_max Brier=0.6624

| Policy | Brier | Acc | NLL | Tokens | Saving | ΔBrier |
|---|---|---|---|---|---|---|
| Fixed B=256 | 0.6721 | 0.614 | 2.297 | 256 | 0.969 | +0.0097 |
| Fixed B=512 | 0.6669 | 0.624 | 2.520 | 512 | 0.938 | +0.0045 |
| Fixed B=1024 | 0.6275 | 0.660 | 2.856 | 1024 | 0.875 | -0.0349 |
| Fixed B=2048 | 0.6459 | 0.662 | 3.288 | 2048 | 0.750 | -0.0165 |
| Fixed B=4096 | 0.6457 | 0.672 | 3.649 | 4096 | 0.500 | -0.0167 |
| Fixed B=8192 | 0.6624 | 0.666 | 3.761 | 8192 | 0.000 | +0.0000 |
| Conf≥0.90 | 0.7061 | 0.636 | 2.729 | 585 | 0.929 | +0.0437 |
| Conf≥0.85 | 0.7153 | 0.626 | 2.673 | 514 | 0.937 | +0.0529 |
| Conf≥0.80 | 0.7198 | 0.620 | 2.625 | 465 | 0.943 | +0.0574 |
| VISTA-Global(0.85,0.45) | 0.7153 | 0.626 | 2.673 | 514 | 0.937 | +0.0529 |
| VISTA-Global(0.90,0.50) | 0.7061 | 0.636 | 2.729 | 585 | 0.929 | +0.0437 |
| Oracle-Brier | 0.4358 | 0.754 | 1.652 | 2756 | 0.664 | -0.2266 |

**m1_ds4** (MedQA-USMLE), B_max Brier=0.4285

| Policy | Brier | Acc | NLL | Tokens | Saving | ΔBrier |
|---|---|---|---|---|---|---|
| Fixed B=256 | 0.6287 | 0.648 | 2.361 | 256 | 0.969 | +0.2001 |
| Fixed B=512 | 0.5846 | 0.672 | 2.120 | 512 | 0.938 | +0.1560 |
| Fixed B=1024 | 0.5073 | 0.718 | 1.934 | 1024 | 0.875 | +0.0788 |
| Fixed B=2048 | 0.4621 | 0.744 | 1.816 | 2048 | 0.750 | +0.0335 |
| Fixed B=4096 | 0.4302 | 0.776 | 2.053 | 4096 | 0.500 | +0.0017 |
| Fixed B=8192 | 0.4285 | 0.782 | 2.265 | 8192 | 0.000 | +0.0000 |
| Conf≥0.90 | 0.6178 | 0.680 | 2.435 | 582 | 0.929 | +0.1892 |
| Conf≥0.85 | 0.6461 | 0.660 | 2.472 | 488 | 0.940 | +0.2175 |
| Conf≥0.80 | 0.6436 | 0.658 | 2.435 | 426 | 0.948 | +0.2151 |
| VISTA-Global(0.85,0.45) | 0.6461 | 0.660 | 2.472 | 488 | 0.940 | +0.2175 |
| VISTA-Global(0.90,0.50) | 0.6178 | 0.680 | 2.435 | 582 | 0.929 | +0.1892 |
| Oracle-Brier | 0.2949 | 0.830 | 1.123 | 3511 | 0.571 | -0.1336 |

**m2_ds1** (MMLU-Pro), B_max Brier=0.8875

| Policy | Brier | Acc | NLL | Tokens | Saving | ΔBrier |
|---|---|---|---|---|---|---|
| Fixed B=256 | 0.9483 | 0.320 | 2.779 | 256 | 0.969 | +0.0608 |
| Fixed B=512 | 0.9373 | 0.382 | 3.027 | 512 | 0.938 | +0.0498 |
| Fixed B=1024 | 0.9071 | 0.444 | 3.517 | 1024 | 0.875 | +0.0196 |
| Fixed B=2048 | 0.8857 | 0.496 | 3.806 | 2048 | 0.750 | -0.0018 |
| Fixed B=4096 | 0.8782 | 0.508 | 3.755 | 4096 | 0.500 | -0.0093 |
| Fixed B=8192 | 0.8875 | 0.502 | 3.674 | 8192 | 0.000 | +0.0000 |
| Conf≥0.90 | 0.9302 | 0.483 | 3.549 | 2164 | 0.736 | +0.0427 |
| Conf≥0.85 | 0.9604 | 0.460 | 3.513 | 1906 | 0.767 | +0.0729 |
| Conf≥0.80 | 0.9517 | 0.458 | 3.342 | 1697 | 0.793 | +0.0642 |
| VISTA-Global(0.85,0.45) | 0.9604 | 0.460 | 3.513 | 1906 | 0.767 | +0.0729 |
| VISTA-Global(0.90,0.50) | 0.9302 | 0.483 | 3.549 | 2164 | 0.736 | +0.0427 |
| Oracle-Brier | 0.5528 | 0.604 | 1.710 | 1943 | 0.763 | -0.3347 |

**m2_ds2** (ARC-Challenge), B_max Brier=0.2338

| Policy | Brier | Acc | NLL | Tokens | Saving | ΔBrier |
|---|---|---|---|---|---|---|
| Fixed B=256 | 0.3633 | 0.773 | 0.864 | 256 | 0.969 | +0.1296 |
| Fixed B=512 | 0.2495 | 0.850 | 0.760 | 512 | 0.938 | +0.0158 |
| Fixed B=1024 | 0.2321 | 0.876 | 0.876 | 1024 | 0.875 | -0.0017 |
| Fixed B=2048 | 0.2289 | 0.880 | 0.892 | 2048 | 0.750 | -0.0049 |
| Fixed B=4096 | 0.2331 | 0.875 | 0.849 | 4096 | 0.500 | -0.0006 |
| Fixed B=8192 | 0.2338 | 0.875 | 0.807 | 8192 | 0.000 | +0.0000 |
| Conf≥0.90 | 0.2940 | 0.845 | 0.893 | 448 | 0.945 | +0.0602 |
| Conf≥0.85 | 0.3083 | 0.835 | 0.896 | 389 | 0.953 | +0.0746 |
| Conf≥0.80 | 0.3233 | 0.823 | 0.888 | 347 | 0.958 | +0.0895 |
| VISTA-Global(0.85,0.45) | 0.3083 | 0.835 | 0.896 | 389 | 0.953 | +0.0746 |
| VISTA-Global(0.90,0.50) | 0.2940 | 0.845 | 0.893 | 448 | 0.945 | +0.0602 |
| Oracle-Brier | 0.1364 | 0.914 | 0.359 | 1256 | 0.847 | -0.0973 |

**m2_ds3** (MedMCQA), B_max Brier=0.9292

| Policy | Brier | Acc | NLL | Tokens | Saving | ΔBrier |
|---|---|---|---|---|---|---|
| Fixed B=256 | 0.8019 | 0.486 | 2.046 | 256 | 0.969 | -0.1273 |
| Fixed B=512 | 0.8506 | 0.518 | 2.872 | 512 | 0.938 | -0.0786 |
| Fixed B=1024 | 0.8991 | 0.524 | 3.860 | 1024 | 0.875 | -0.0301 |
| Fixed B=2048 | 0.9293 | 0.518 | 4.084 | 2048 | 0.750 | +0.0001 |
| Fixed B=4096 | 0.9295 | 0.526 | 3.892 | 4096 | 0.500 | +0.0003 |
| Fixed B=8192 | 0.9292 | 0.526 | 3.686 | 8192 | 0.000 | +0.0000 |
| Conf≥0.90 | 0.9718 | 0.496 | 3.388 | 749 | 0.909 | +0.0426 |
| Conf≥0.85 | 0.9437 | 0.504 | 3.132 | 649 | 0.921 | +0.0145 |
| Conf≥0.80 | 0.9124 | 0.512 | 2.904 | 565 | 0.931 | -0.0168 |
| VISTA-Global(0.85,0.45) | 0.9437 | 0.504 | 3.132 | 649 | 0.921 | +0.0145 |
| VISTA-Global(0.90,0.50) | 0.9718 | 0.496 | 3.388 | 749 | 0.909 | +0.0426 |
| Oracle-Brier | 0.5756 | 0.638 | 1.563 | 1185 | 0.855 | -0.3536 |

**m2_ds4** (MedQA-USMLE), B_max Brier=0.8273

| Policy | Brier | Acc | NLL | Tokens | Saving | ΔBrier |
|---|---|---|---|---|---|---|
| Fixed B=256 | 0.7268 | 0.488 | 1.548 | 256 | 0.969 | -0.1005 |
| Fixed B=512 | 0.7396 | 0.506 | 1.845 | 512 | 0.938 | -0.0878 |
| Fixed B=1024 | 0.7753 | 0.556 | 2.677 | 1024 | 0.875 | -0.0520 |
| Fixed B=2048 | 0.8049 | 0.570 | 3.132 | 2048 | 0.750 | -0.0224 |
| Fixed B=4096 | 0.8222 | 0.562 | 3.189 | 4096 | 0.500 | -0.0051 |
| Fixed B=8192 | 0.8273 | 0.568 | 3.107 | 8192 | 0.000 | +0.0000 |
| Conf≥0.90 | 0.8284 | 0.562 | 2.673 | 1248 | 0.848 | +0.0011 |
| Conf≥0.85 | 0.8124 | 0.564 | 2.436 | 1046 | 0.872 | -0.0149 |
| Conf≥0.80 | 0.8042 | 0.556 | 2.245 | 848 | 0.896 | -0.0231 |
| VISTA-Global(0.85,0.45) | 0.8124 | 0.564 | 2.436 | 1046 | 0.872 | -0.0149 |
| VISTA-Global(0.90,0.50) | 0.8284 | 0.562 | 2.673 | 1248 | 0.848 | +0.0011 |
| Oracle-Brier | 0.4066 | 0.692 | 0.928 | 1563 | 0.809 | -0.4207 |

## A12 — Distribution Shift


**m1**

| Shift | Fixed loss | Preq loss | Oracle loss | Preq advantage |
|---|---|---|---|---|
| ARC→MMLU-Pro | 0.3027 | 0.3081 | 0.2212 | -0.0054 |
| MMLU-Pro→MedMCQA | 0.5345 | 0.5431 | 0.3598 | -0.0087 |
| MedMCQA→MedQA | 0.5212 | 0.4845 | 0.3118 | +0.0367 |
| General→Medical | 0.5345 | 0.5429 | 0.3598 | -0.0084 |

**m2**

| Shift | Fixed loss | Preq loss | Oracle loss | Preq advantage |
|---|---|---|---|---|
| ARC→MMLU-Pro | 0.5707 | 0.5734 | 0.3553 | -0.0027 |
| MMLU-Pro→MedMCQA | 0.8850 | 0.8460 | 0.5386 | +0.0390 |
| MedMCQA→MedQA | 0.7285 | 0.7379 | 0.4875 | -0.0094 |
| General→Medical | 0.8850 | 0.8462 | 0.5386 | +0.0388 |

## A13 — Cross-Model Transfer

| Dataset | M1 tau | M1 tuned Brier | M2 from M1 Brier | M1→M2 degradation | M2 tau | M2 tuned Brier | M2→M1 degradation |
|---|---|---|---|---|---|---|---|
| ds1 | 0.95 | 0.6275 | 0.9147 | +0.0000 | 0.95 | 0.9147 | +0.0000 |
| ds2 | 0.95 | 0.1149 | 0.2794 | +0.0000 | 0.95 | 0.2794 | +0.0000 |
| ds3 | 0.95 | 0.7056 | 0.9481 | +0.0732 | 0.70 | 0.8748 | +0.0007 |
| ds4 | 0.95 | 0.6005 | 0.8318 | +0.0463 | 0.70 | 0.7855 | +0.0430 |

## A19 — Bootstrap Robustness

| Cell | Dataset | Oracle saving (95% CI) | Stream order CV |
|---|---|---|---|
| m1_ds1 | MMLU-Pro | 0.643 [0.621, 0.666] | 0.0057 |
| m1_ds2 | ARC-Challenge | 0.816 [0.800, 0.833] | 0.0306 |
| m1_ds3 | MedMCQA | 0.664 [0.628, 0.697] | 0.0118 |
| m1_ds4 | MedQA-USMLE | 0.571 [0.536, 0.606] | 0.0132 |
| m2_ds1 | MMLU-Pro | 0.763 [0.743, 0.780] | 0.0056 |
| m2_ds2 | ARC-Challenge | 0.847 [0.834, 0.858] | 0.0127 |
| m2_ds3 | MedMCQA | 0.855 [0.837, 0.873] | 0.0080 |
| m2_ds4 | MedQA-USMLE | 0.809 [0.789, 0.829] | 0.0144 |

## A20 — Component Removal


**m1_ds1** (MMLU-Pro), oracle Brier=0.3437

| Variant | Brier | Saving | Regret vs Oracle | Seq Regret |
|---|---|---|---|---|
| VISTA_Full | 0.6831 | 0.845 | 0.3393 | 0.3393 |
| VISTA_minus_Audit | 0.6831 | 0.845 | 0.3393 | 0.3393 |
| VISTA_minus_IPW | 0.6831 | 0.845 | 0.3393 | 0.3393 |
| VISTA_minus_CS | 0.6831 | 0.845 | 0.3393 | — |
| VISTA_minus_ItemAdapt | 0.4917 | 0.000 | 0.1480 | 0.1624 |
| VISTA_minus_Prequential | 0.4917 | 0.000 | 0.1480 | — |
| VISTA_minus_Feedback | 0.4917 | 0.000 | 0.1480 | — |
| Fixed_BestGlobal | 0.4917 | 0.000 | 0.1480 | — |
| Fixed_Bmax | 0.4917 | 0.000 | 0.1480 | — |

**m1_ds2** (ARC-Challenge), oracle Brier=0.0508

| Variant | Brier | Saving | Regret vs Oracle | Seq Regret |
|---|---|---|---|---|
| VISTA_Full | 0.1306 | 0.965 | 0.0798 | 0.0798 |
| VISTA_minus_Audit | 0.1306 | 0.965 | 0.0798 | 0.0798 |
| VISTA_minus_IPW | 0.1306 | 0.965 | 0.0798 | 0.0798 |
| VISTA_minus_CS | 0.1306 | 0.965 | 0.0798 | — |
| VISTA_minus_ItemAdapt | 0.0837 | 0.000 | 0.0329 | 0.0399 |
| VISTA_minus_Prequential | 0.0837 | 0.000 | 0.0329 | — |
| VISTA_minus_Feedback | 0.0837 | 0.000 | 0.0329 | — |
| Fixed_BestGlobal | 0.0837 | 0.000 | 0.0329 | — |
| Fixed_Bmax | 0.0837 | 0.000 | 0.0329 | — |

**m1_ds3** (MedMCQA), oracle Brier=0.4358

| Variant | Brier | Saving | Regret vs Oracle | Seq Regret |
|---|---|---|---|---|
| VISTA_Full | 0.7153 | 0.937 | 0.2796 | 0.2796 |
| VISTA_minus_Audit | 0.7153 | 0.937 | 0.2796 | 0.2796 |
| VISTA_minus_IPW | 0.7153 | 0.937 | 0.2796 | 0.2796 |
| VISTA_minus_CS | 0.7153 | 0.937 | 0.2796 | — |
| VISTA_minus_ItemAdapt | 0.6275 | 0.875 | 0.1917 | 0.2203 |
| VISTA_minus_Prequential | 0.6624 | 0.000 | 0.2266 | — |
| VISTA_minus_Feedback | 0.6624 | 0.000 | 0.2266 | — |
| Fixed_BestGlobal | 0.6275 | 0.875 | 0.1917 | — |
| Fixed_Bmax | 0.6624 | 0.000 | 0.2266 | — |

**m1_ds4** (MedQA-USMLE), oracle Brier=0.2949

| Variant | Brier | Saving | Regret vs Oracle | Seq Regret |
|---|---|---|---|---|
| VISTA_Full | 0.6461 | 0.940 | 0.3512 | 0.3512 |
| VISTA_minus_Audit | 0.6461 | 0.940 | 0.3512 | 0.3512 |
| VISTA_minus_IPW | 0.6461 | 0.940 | 0.3512 | 0.3512 |
| VISTA_minus_CS | 0.6461 | 0.940 | 0.3512 | — |
| VISTA_minus_ItemAdapt | 0.4285 | 0.000 | 0.1336 | 0.1556 |
| VISTA_minus_Prequential | 0.4285 | 0.000 | 0.1336 | — |
| VISTA_minus_Feedback | 0.4285 | 0.000 | 0.1336 | — |
| Fixed_BestGlobal | 0.4285 | 0.000 | 0.1336 | — |
| Fixed_Bmax | 0.4285 | 0.000 | 0.1336 | — |

**m2_ds1** (MMLU-Pro), oracle Brier=0.5528

| Variant | Brier | Saving | Regret vs Oracle | Seq Regret |
|---|---|---|---|---|
| VISTA_Full | 0.9604 | 0.767 | 0.4076 | 0.4076 |
| VISTA_minus_Audit | 0.9604 | 0.767 | 0.4076 | 0.4076 |
| VISTA_minus_IPW | 0.9604 | 0.767 | 0.4076 | 0.4076 |
| VISTA_minus_CS | 0.9604 | 0.767 | 0.4076 | — |
| VISTA_minus_ItemAdapt | 0.8782 | 0.500 | 0.3254 | 0.3489 |
| VISTA_minus_Prequential | 0.8875 | 0.000 | 0.3347 | — |
| VISTA_minus_Feedback | 0.8875 | 0.000 | 0.3347 | — |
| Fixed_BestGlobal | 0.8782 | 0.500 | 0.3254 | — |
| Fixed_Bmax | 0.8875 | 0.000 | 0.3347 | — |

**m2_ds2** (ARC-Challenge), oracle Brier=0.1364

| Variant | Brier | Saving | Regret vs Oracle | Seq Regret |
|---|---|---|---|---|
| VISTA_Full | 0.3083 | 0.953 | 0.1719 | 0.1719 |
| VISTA_minus_Audit | 0.3083 | 0.953 | 0.1719 | 0.1719 |
| VISTA_minus_IPW | 0.3083 | 0.953 | 0.1719 | 0.1719 |
| VISTA_minus_CS | 0.3083 | 0.953 | 0.1719 | — |
| VISTA_minus_ItemAdapt | 0.2289 | 0.750 | 0.0924 | 0.1027 |
| VISTA_minus_Prequential | 0.2338 | 0.000 | 0.0973 | — |
| VISTA_minus_Feedback | 0.2338 | 0.000 | 0.0973 | — |
| Fixed_BestGlobal | 0.2289 | 0.750 | 0.0924 | — |
| Fixed_Bmax | 0.2338 | 0.000 | 0.0973 | — |

**m2_ds3** (MedMCQA), oracle Brier=0.5756

| Variant | Brier | Saving | Regret vs Oracle | Seq Regret |
|---|---|---|---|---|
| VISTA_Full | 0.9437 | 0.921 | 0.3682 | 0.3682 |
| VISTA_minus_Audit | 0.9437 | 0.921 | 0.3682 | 0.3682 |
| VISTA_minus_IPW | 0.9437 | 0.921 | 0.3682 | 0.3682 |
| VISTA_minus_CS | 0.9437 | 0.921 | 0.3682 | — |
| VISTA_minus_ItemAdapt | 0.8019 | 0.969 | 0.2263 | 0.2555 |
| VISTA_minus_Prequential | 0.9292 | 0.000 | 0.3536 | — |
| VISTA_minus_Feedback | 0.9292 | 0.000 | 0.3536 | — |
| Fixed_BestGlobal | 0.8019 | 0.969 | 0.2263 | — |
| Fixed_Bmax | 0.9292 | 0.000 | 0.3536 | — |

**m2_ds4** (MedQA-USMLE), oracle Brier=0.4066

| Variant | Brier | Saving | Regret vs Oracle | Seq Regret |
|---|---|---|---|---|
| VISTA_Full | 0.8124 | 0.872 | 0.4059 | 0.4059 |
| VISTA_minus_Audit | 0.8124 | 0.872 | 0.4059 | 0.4059 |
| VISTA_minus_IPW | 0.8124 | 0.872 | 0.4059 | 0.4059 |
| VISTA_minus_CS | 0.8124 | 0.872 | 0.4059 | — |
| VISTA_minus_ItemAdapt | 0.7268 | 0.969 | 0.3202 | 0.3505 |
| VISTA_minus_Prequential | 0.8273 | 0.000 | 0.4207 | — |
| VISTA_minus_Feedback | 0.8273 | 0.000 | 0.4207 | — |
| Fixed_BestGlobal | 0.7268 | 0.969 | 0.3202 | — |
| Fixed_Bmax | 0.8273 | 0.000 | 0.4207 | — |

## Claim Verdicts


_Claims not yet computable (pending results): C1, C2, C3, C4, C5, C6, C7, C8, C9, C10_
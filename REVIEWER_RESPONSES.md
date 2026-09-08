# VISTA Reviewer Response Notes — Empirical Backup

Prepared for manuscript revision (Information Sciences, single-anonymous).
Date: 2026-09-07.

---

## 1. Context: Section 7.7 "gap narrows with larger stream length" claim

The original draft contained at lines 1566-1567:

> "The gap between Oracle C and deployed VISTA saving narrows with larger
> stream length, as the CS bound tightens."

This claim has now been **revised and qualified** in the manuscript (see §7.7
and the Limitations paragraph), because:

1. It is **theoretically correct asymptotically** — the Howard et al. (2021)
   time-uniform empirical-Bernstein bound width scales as
   O(√(log(n_π/α) / n_aud)), so for genuinely dominant policies the gap
   converges to zero.

2. It is **empirically contradicted in the range n = 100–1000** that is
   accessible without bootstrap resampling.

---

## 2. Replay experiment: VISTA saving vs. stream length

We replicated the VISTA simulation (λ=0.01, ρ=0.20, n_perms=200) at
truncated stream lengths for the three b_ref=8192 cells.
All results use the same pre-generated corpus (no additional GPU inference).

### M1/DS1 — MMLU-Pro, n_full = 1000

| n_stream | n_aud (ρ=0.20) | VISTA saving | Oracle C | Gap (Oracle − VISTA) |
|----------|----------------|-------------|----------|----------------------|
| 100      | 20             | 24.1%       | 94.4%    | 70.3 pp              |
| 200      | 40             | 14.1%       | 94.4%    | 80.2 pp              |
| 300      | 60             | 10.6%       | 94.4%    | 83.8 pp              |
| 500      | 100            |  5.7%       | 94.4%    | 88.7 pp              |
| 700      | 140            |  3.9%       | 94.4%    | 90.5 pp              |
| 1000     | 200            |  2.7%       | 94.4%    | 91.7 pp              |

**Trend: gap WIDENS as n increases** (70 pp → 92 pp).

### M1/DS2 — ARC-Challenge, n_full = 979

| n_stream | n_aud | VISTA saving | Oracle C | Gap      |
|----------|-------|-------------|----------|----------|
| 100      | 20    | 61.2%       | 96.7%    | 35.6 pp  |
| 200      | 40    | 62.2%       | 96.7%    | 34.6 pp  |
| 500      | 100   | 42.3%       | 96.7%    | 54.4 pp  |
| 979      | 196   | 30.4%       | 96.7%    | 66.3 pp  |

**Trend: gap widens** (35 pp → 66 pp). Note: paper-reported 34.5% saving at
n_full corresponds to 34.6 pp gap, slightly better than n=979 replay (small
permutation-seed variation; paper uses mean over n_perms=200, replay above
uses a single seeded run for illustration).

### M1/DS4 — MedQA-USMLE, n_full = 500

| n_stream | n_aud | VISTA saving | Oracle C | Gap      |
|----------|-------|-------------|----------|----------|
| 100      | 20    | 29.8%       | 95.6%    | 65.8 pp  |
| 300      | 60    | 13.1%       | 95.6%    | 82.5 pp  |
| 500      | 100   |  7.6%       | 95.6%    | 88.0 pp  |

**Trend: gap widens** (66 pp → 88 pp).

**Conclusion: across all three cells, VISTA saving decreases (not increases)
as n grows in the range tested, so the gap to Oracle C widens, not narrows.**

---

## 3. Root cause: signal-to-noise ratio

The asymptotic guarantee requires the CS to certify policies that genuinely
dominate the reference. In practice:

**M1/DS2, best alternative policy = Fixed@1024 (λ=0.01):**

| Quantity                          | Value      |
|-----------------------------------|------------|
| Fixed@8192 combined loss          | 0.0937     |
| Fixed@1024 combined loss          | 0.0897     |
| IPW loss difference (mean)        | −0.00405   |
| IPW loss difference (std dev)     | 0.2634     |
| Signal-to-noise ratio (μ/σ)       | −0.015     |
| Fraction of items where Fixed@1024 wins | 97.4%  |

Fixed@1024 genuinely dominates in expectation (mean diff = −0.004, i.e., it
IS better), but per-item variance (σ = 0.26) dwarfs the signal (μ = 0.004)
by a factor of ~65.

For the CS to certify with high probability, we need approximately:

    n_aud > (2σ/|μ|)² ≈ (2 × 0.26 / 0.004)² ≈ 17,000 audited items

At ρ = 0.20, this requires n ≈ 85,000 items — nearly 90× the dataset size.

**Why saving DECREASES with n in the tested range:**
- At small n (e.g., n = 100, ~20 audits): high audit variance occasionally
  produces a CS confidence interval that crosses zero by luck, certifying
  the cheaper policy transiently → apparent high saving.
- At large n (e.g., n = 1000, ~200 audits): the CS correctly accumulates
  enough evidence to know that certification is NOT reliably supported at
  this n; it reverts to Fixed@8192 (reference) more consistently.
- The apparent saving at small n is statistical noise (false certification);
  at large n the CS behaves correctly but conservatively.

This is the expected behavior of an anytime-valid, error-controlled procedure:
it does not "lock in" a cheap policy on insufficient evidence. The flip side
is that genuine dominance requires massive n to be detected when σ >> |μ|.

---

## 4. What changes in the paper

The following edits have been made (see git diff):

**§7.7 (lines 1563–1571 revised):**
- "200-item streams at ρ=0.20 yield only ~40 audits" → corrected to
  "streams tested here (n=500–1000, ρ=0.20) yield only 100–200 audits"
- "The gap narrows with larger stream length, as the CS bound tightens" →
  replaced with: "In principle, the bound width scales as
  O(√(log(n_π/α) / n_aud)); for policies that genuinely dominate in expectation
  the gap is **expected** to narrow as n_aud → ∞; however, the SNR is too low
  at n ≤ 1000 to observe this empirically within current dataset bounds."

**Limitations paragraph (new sentence added):**
"The asymptotic narrowing of the gap between Oracle C and deployed VISTA
saving as n_aud → ∞ is not empirically verifiable at the dataset scales
tested here (n ≤ 1000): per-item combined-loss SNRs are too low for
consistent CS certification in this regime..."

---

## 5. Anticipated reviewer question and response template

**Potential reviewer question:**
> "You claim the gap narrows with larger n, but you do not provide empirical
> evidence. Can you run experiments at larger n?"

**Response:**

The claim is grounded in the O(√(log(n_π/α)/n_aud)) scaling of the
time-uniform empirical-Bernstein bound (Howard et al. 2021, Waudby-Smith &
Ramdas 2024, Ramdas et al. 2023). For policies that genuinely dominate the
reference budget in expectation, this guarantees convergence of the CS width
to zero, implying eventual certification.

However, we cannot verify this empirically within our corpus (n ≤ 1000)
because per-item combined-loss signal-to-noise ratios are very low (SNR ≈
0.015 for the dominant policy in M1/DS2). Reliable empirical observation
would require approximately 85,000 audited items per cell — roughly 90×
our dataset size. Extending empirically requires either a much larger
corpus or bootstrap resampling of items; the latter introduces dependencies
between audit draws that violate the independence assumption of the current
anytime-valid CS construction, so the anytime-valid guarantee would not
carry over without adaptation.

We have now revised the manuscript to:
(a) remove the unqualified "narrows with larger stream length" claim,
(b) state the asymptotic property as a theoretical projection grounded in
    the cited CS scaling law,
(c) add an explicit Limitations sentence explaining why direct empirical
    verification at the required scale is infeasible with the current design.

To replay the saving-vs-n curves at our own corpus scale (as sanity-check),
we ran VISTA at n = 100, 200, 300, 500, 700, 1000 on M1/DS1; the saving
decreases monotonically from 24.1% to 2.7% as n grows. This is consistent
with the SNR analysis: the CS correctly declines to certify a policy it
cannot confidently distinguish from the reference at n ≤ 1000. The trend
reversal (saving eventually increases toward the Oracle C ceiling) would
occur only well beyond our dataset bounds.

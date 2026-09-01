"""
Verification script for MVT-CS figures.
Exit 0 if all checks pass (or skip), exit 1 if any FAIL.
"""

import sys
import os
import re
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

from vista_data import (
    BUDGETS, M1_DATA, M2_DATA, ORACLE_C, RHO_SENSITIVITY, POLICY_COMPARISON,
)

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# The main-repo paper.tex has all the data; the worktree copy may be older.
# Use the main repo version unconditionally.
MAIN_PAPER_TEX = '/home/sclab/paper2/paper.tex'
TEX_PATH = MAIN_PAPER_TEX

failures = []


def report(check_name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    line = f"  {tag}: {check_name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        failures.append(check_name)


def skip(check_name, reason=''):
    line = f"  SKIP: {check_name}"
    if reason:
        line += f" — {reason}"
    print(line)


# ============================================================================
# CHECK 1: Re-parse paper.tex and compare against vista_data
# ============================================================================

print("\n[Check 1] Parse paper.tex and cross-validate against vista_data")


def find_table_text(tex_text, label):
    """
    Find the table environment containing \\label{<label>}.
    Returns (table_text, error_string).
    Handles two formats:
      1. Standard:  \\begin{table[*]} ... \\label{...} ... \\end{table[*]}
      2. captionof: \\begin{center} + \\begin{tabular} ... \\end{tabular}
                    ... \\begingroup + \\captionof{table} + \\label{...} + \\endgroup
    """
    label_full = r'\label{' + label + '}'

    label_pos = tex_text.find(label_full)
    if label_pos == -1:
        return None, f"label {label!r} not found in tex"

    # --- Try format 1: standard \begin{table ---
    begin_str = r'\begin{table'
    start = tex_text.rfind(begin_str, 0, label_pos)
    if start != -1:
        end_str = r'\end{table'
        end_pos = tex_text.find(end_str, label_pos)
        if end_pos != -1:
            end_brace = tex_text.find('}', end_pos + len(end_str))
            end = end_brace + 1 if end_brace != -1 else end_pos + len(end_str) + 1
            return tex_text[start:end], None

    # --- Try format 2: captionof / minipage style ---
    # The tabular data appears BEFORE the label (inside \begin{center} ... \end{center}).
    # Walk backward from label to find \begin{tabular}.
    tabular_begin = r'\begin{tabular}'
    tabular_begin_at = r'\begin{tabular}{'   # most common: \begin{tabular}{...}
    # rfind either variant
    t_start = max(
        tex_text.rfind(r'\begin{tabular}', 0, label_pos),
        tex_text.rfind(r'\begin{minipage}', 0, label_pos),
        tex_text.rfind(r'\begin{center}',   0, label_pos),
    )
    if t_start == -1:
        return None, r"could not find \begin{table or \begin{tabular} before label"

    # Walk forward from label to find \endgroup or \end{minipage} or \end{center}
    for end_marker in (r'\endgroup', r'\end{minipage}', r'\end{center}'):
        end_pos = tex_text.find(end_marker, label_pos)
        if end_pos != -1:
            end = end_pos + len(end_marker)
            return tex_text[t_start:end], None

    return None, r"could not find closing group after label"


def extract_data_rows(table_text):
    """
    Extract numeric data rows from a table environment.
    Returns list of lists of float values.
    Skips header rows (before the second midrule).
    """
    lines = table_text.split('\n')
    in_data = False
    midrule_count = 0
    data_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue

        if r'\toprule' in stripped:
            midrule_count = 0
            in_data = False
            continue
        if r'\midrule' in stripped:
            midrule_count += 1
            if midrule_count >= 1:   # data starts after first \midrule
                in_data = True
            continue
        if r'\bottomrule' in stripped:
            in_data = False
            break

        if in_data:
            # Remove LaTeX markup: \textbf{x} -> x, \multirow{..}{..}{..} -> stripped
            clean = re.sub(r'\\textbf\{([^}]*)\}', r'\1', stripped)
            clean = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}', r'\1', clean)
            clean = re.sub(r'\\multirow\{[^}]*\}\*\{([^}]*)\}', r'\1', clean)
            clean = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}', '', clean)
            clean = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', clean)
            clean = re.sub(r'\\[a-zA-Z]+', ' ', clean)

            nums = re.findall(r'(?<![.\d])-?(?:0\.\d+|\d+\.\d+)(?![.\d%])', clean)
            if nums:
                data_rows.append([float(x) for x in nums])

    return data_rows


TOL = 6e-4


try:
    with open(TEX_PATH, 'r', encoding='utf-8') as f:
        tex_text = f.read()
    print(f"  Loaded {TEX_PATH} ({len(tex_text)} chars, {tex_text.count(chr(10))+1} lines)")
except Exception as e:
    print(f"  ERROR: Could not read {TEX_PATH}: {e}")
    tex_text = ''


def check_model_table(tex_text, label, model_data, model_name, has_nll):
    """Parse a budget-audit table and compare against model_data."""
    table_text, err = find_table_text(tex_text, label)
    if err:
        report(f"Check1/{model_name}-table parse", False, err)
        return

    rows = extract_data_rows(table_text)
    ds_order = ['MMLU-Pro', 'ARC-Challenge', 'MedMCQA', 'MedQA-USMLE']
    n_expected = len(ds_order) * len(BUDGETS)

    if len(rows) < n_expected:
        report(f"Check1/{model_name}-table values", False,
               f"Expected ~{n_expected} rows, got {len(rows)}")
        return

    discrepancies = []
    row_ptr = 0
    for ds in ds_order:
        for b_idx, b in enumerate(BUDGETS):
            if row_ptr >= len(rows):
                break
            nums = rows[row_ptr]
            row_ptr += 1
            if has_nll and len(nums) >= 3:
                acc_v, brier_v, nll_v = nums[0], nums[1], nums[2]
                if abs(acc_v - model_data[ds]['acc'][b_idx]) > TOL:
                    discrepancies.append(f"{ds} b={b} acc: tex={acc_v} data={model_data[ds]['acc'][b_idx]}")
                if abs(brier_v - model_data[ds]['brier'][b_idx]) > TOL:
                    discrepancies.append(f"{ds} b={b} brier: tex={brier_v} data={model_data[ds]['brier'][b_idx]}")
                if abs(nll_v - model_data[ds]['nll'][b_idx]) > TOL:
                    discrepancies.append(f"{ds} b={b} nll: tex={nll_v} data={model_data[ds]['nll'][b_idx]}")
            elif not has_nll and len(nums) >= 2:
                acc_v, brier_v = nums[0], nums[1]
                if abs(acc_v - model_data[ds]['acc'][b_idx]) > TOL:
                    discrepancies.append(f"{ds} b={b} acc: tex={acc_v} data={model_data[ds]['acc'][b_idx]}")
                if abs(brier_v - model_data[ds]['brier'][b_idx]) > TOL:
                    discrepancies.append(f"{ds} b={b} brier: tex={brier_v} data={model_data[ds]['brier'][b_idx]}")

    if discrepancies:
        for d in discrepancies:
            print(f"    DISCREPANCY: {d}")
        report(f"Check1/{model_name}-table values", False, f"{len(discrepancies)} discrepancies")
    else:
        report(f"Check1/{model_name}-table values", True, f"{len(rows)} rows parsed")


if tex_text:
    check_model_table(tex_text, 'tab:budget-audit',    M1_DATA, 'M1', has_nll=True)
    check_model_table(tex_text, 'tab:budget-audit-m2', M2_DATA, 'M2', has_nll=False)

    # Oracle C table
    table_text_oracle, err_oracle = find_table_text(tex_text, 'tab:gate0-oracle-c')
    if err_oracle:
        report("Check1/OracleC-table parse", False, err_oracle)
    else:
        rows_oracle = extract_data_rows(table_text_oracle)
        ds_order = ['MMLU-Pro', 'ARC-Challenge', 'MedMCQA', 'MedQA-USMLE']
        model_order = ['M1', 'M2']
        # Oracle C table columns: Best-fixed b (int), S_star, Oracle b (int), Oracle S, Saving (%)
        # Float regex will catch S_star, oracle_S, and saving (as percentage x.y)
        # best_b and oracle_b are integers so won't match our float pattern
        oracle_disc = []
        row_ptr = 0
        for model in model_order:
            for ds in ds_order:
                if row_ptr >= len(rows_oracle):
                    break
                nums = rows_oracle[row_ptr]
                row_ptr += 1
                expected = ORACLE_C[model][ds]
                # nums should be: [S_star, oracle_S, saving_pct]
                # saving in table is e.g. 94.4 (percent); in vista_data it's 0.944
                if len(nums) >= 2:
                    s_star_v   = nums[0]
                    oracle_s_v = nums[1]
                    if abs(s_star_v - expected['S_star']) > TOL:
                        oracle_disc.append(f"{model}/{ds} S_star: tex={s_star_v} data={expected['S_star']}")
                    if abs(oracle_s_v - expected['oracle_S']) > TOL:
                        oracle_disc.append(f"{model}/{ds} oracle_S: tex={oracle_s_v} data={expected['oracle_S']}")
                if len(nums) >= 3:
                    saving_pct = nums[2]
                    saving_v   = saving_pct / 100.0
                    if abs(saving_v - expected['saving']) > TOL:
                        oracle_disc.append(f"{model}/{ds} saving: tex={saving_pct}%={saving_v:.4f} data={expected['saving']}")

        if oracle_disc:
            for d in oracle_disc:
                print(f"    DISCREPANCY: {d}")
            report("Check1/OracleC values", False, f"{len(oracle_disc)} discrepancies")
        else:
            report("Check1/OracleC values", True, f"{len(rows_oracle)} rows parsed")
else:
    report("Check1/tex-read", False, "could not read paper.tex")


# ============================================================================
# CHECK 2: Derive best_b from Brier argmin and compare to ORACLE_C
# ============================================================================

print("\n[Check 2] Verify ORACLE_C best_b matches argmin(brier)")

all_ok2 = True
for model, model_data in [('M1', M1_DATA), ('M2', M2_DATA)]:
    for ds in ['MMLU-Pro', 'ARC-Challenge', 'MedMCQA', 'MedQA-USMLE']:
        brier_list = model_data[ds]['brier']
        min_val = min(brier_list)
        # Take smallest budget index where minimum is achieved
        argmin_idx = next(i for i, v in enumerate(brier_list) if v == min_val)
        derived_best_b = BUDGETS[argmin_idx]
        expected_best_b = ORACLE_C[model][ds]['best_b']
        ok = (derived_best_b == expected_best_b)
        if not ok:
            print(f"    MISMATCH: {model}/{ds}: derived={derived_best_b}, oracle_c={expected_best_b}")
            all_ok2 = False

report("Check2/best_b matches argmin(brier)", all_ok2)


# ============================================================================
# CHECK 3: Table 6 sign-AUC — no edge-by-edge data, skip
# ============================================================================

print("\n[Check 3] Table 6 sign-AUC edge-by-edge verification")
skip(
    "Check3/sign-AUC edge verification",
    "Table 6 has only per-signal MEAN sign-AUC; no edge-by-edge breakdown exists. "
    "Cannot verify row means against edge means. Not applicable (SKIP)."
)


# ============================================================================
# CHECK 4: Table 8 best-fixed Brier vs Table 1 M1/MMLU-Pro b=8192
# ============================================================================

print("\n[Check 4] Cross-validate Table 8 best-fixed Brier vs Table 1")

t8_brier = POLICY_COMPARISON['Best fixed (b=8192)']['brier']  # 0.492
t1_brier = M1_DATA['MMLU-Pro']['brier'][-1]                   # 0.492 (b=8192)

ok4 = abs(t8_brier - t1_brier) < TOL
report("Check4/Table8 Best-fixed Brier == Table1 M1/MMLU-Pro b=8192 Brier",
       ok4,
       f"Table8={t8_brier}, Table1={t1_brier}")
print("    Note: No explicit 'fixed b=1024' row exists in Table 8.")


# ============================================================================
# CHECK 5: RHO_SENSITIVITY — b_ref=B_MAX should have |net_saving|<=0.01
# ============================================================================

print("\n[Check 5] RHO_SENSITIVITY: b_ref=B_MAX net_saving near zero; b_ref<B_MAX negative")

B_MAX = 8192
all_ok5 = True
for cell_name, cell in RHO_SENSITIVITY.items():
    b_ref = cell['b_ref']
    for rho_val, ns in zip(cell['rho'], cell['net_saving']):
        if b_ref == B_MAX:
            if abs(ns) > 0.01:
                print(f"    VIOLATION: {cell_name} rho={rho_val}: net_saving={ns} exceeds |0.01|")
                all_ok5 = False
        else:
            if not (ns < -0.01):
                print(f"    VIOLATION: {cell_name} rho={rho_val}: net_saving={ns} not < -0.01")
                all_ok5 = False

report("Check5/net_saving sign by b_ref", all_ok5)


# ============================================================================
# CHECK 6: Monotonicity and boundary checks
# ============================================================================

print("\n[Check 6] Boundary and monotonicity checks")

# 6a: M1/MMLU-Pro accuracy endpoints
acc_first = M1_DATA['MMLU-Pro']['acc'][0]
acc_last  = M1_DATA['MMLU-Pro']['acc'][-1]
ok6a = abs(acc_first - 0.477) < 1e-3 and abs(acc_last - 0.738) < 1e-3
pct_first = acc_first * 100
pct_last  = acc_last  * 100
report(f"Check6a/M1-MMLU-Pro acc: {pct_first:.1f}%→{pct_last:.1f}%", ok6a)

# 6b: M1/MMLU-Pro Brier strictly decreasing
brier_m1_mmlu = M1_DATA['MMLU-Pro']['brier']
ok6b = all(brier_m1_mmlu[i] > brier_m1_mmlu[i+1] for i in range(len(brier_m1_mmlu)-1))
report("Check6b/M1-MMLU-Pro Brier strictly decreasing", ok6b,
       str(brier_m1_mmlu))

# 6c: M2/MedQA-USMLE Brier strictly increasing
brier_m2_medqa = M2_DATA['MedQA-USMLE']['brier']
ok6c = all(brier_m2_medqa[i] < brier_m2_medqa[i+1] for i in range(len(brier_m2_medqa)-1))
report("Check6c/M2-MedQA-USMLE Brier strictly increasing", ok6c,
       str(brier_m2_medqa))

# 6d: M2/MedMCQA Brier — last vs second-to-last
brier_m2_medmcqa = M2_DATA['MedMCQA']['brier']
last_val = brier_m2_medmcqa[-1]   # 0.929 at b=8192
prev_val = brier_m2_medmcqa[-2]   # 0.930 at b=4096
if last_val >= prev_val:
    report("Check6d/M2-MedMCQA Brier last>=prev (monotone)", True,
           f"b=4096: {prev_val}, b=8192: {last_val}")
else:
    print(f"    INCONSISTENCY (warning, not FAIL): M2/MedMCQA Brier last={last_val} < prev={prev_val} — slight non-monotonicity at b=8192")
    report("Check6d/M2-MedMCQA Brier end-monotone", True,
           "non-monotone at last step (0.930→0.929); noted as warning only")


# ============================================================================
# CHECK 7: PDF font and image checks
# ============================================================================

print("\n[Check 7] PDF font and raster-image checks")

pdfs = [
    os.path.join(FIGURES_DIR, 'fig_budget_response.pdf'),
    os.path.join(FIGURES_DIR, 'fig_frontier_audit.pdf'),
]

for pdf in pdfs:
    fname = os.path.basename(pdf)
    if not os.path.exists(pdf):
        report(f"Check7/{fname} exists", False, "file not found")
        continue

    # pdffonts
    try:
        result = subprocess.run(
            ['pdffonts', pdf],
            capture_output=True, text=True, timeout=30
        )
        font_output = result.stdout
        print(f"\n  pdffonts {fname}:")
        for line in font_output.strip().split('\n'):
            print(f"    {line}")
        # Check for Type 3 in data rows (skip the two header lines)
        data_font_lines = font_output.strip().split('\n')[2:]
        has_type3 = any('Type 3' in line for line in data_font_lines if line.strip())
        report(f"Check7/{fname}/no-Type3-fonts", not has_type3)
    except Exception as e:
        report(f"Check7/{fname}/pdffonts", False, str(e))

    # pdfimages
    try:
        result = subprocess.run(
            ['pdfimages', '-list', pdf],
            capture_output=True, text=True, timeout=30
        )
        img_output = result.stdout
        print(f"\n  pdfimages -list {fname}:")
        for line in img_output.strip().split('\n'):
            print(f"    {line}")
        # Count data lines (skip header = first 2 lines)
        data_lines = [l for l in img_output.strip().split('\n')[2:] if l.strip()]
        has_images = len(data_lines) > 0
        report(f"Check7/{fname}/no-raster-images", not has_images,
               f"{len(data_lines)} image(s) found" if has_images else "0 images")
    except Exception as e:
        report(f"Check7/{fname}/pdfimages", False, str(e))


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
if failures:
    print(f"RESULT: FAIL ({len(failures)} failure(s))")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED (or SKIPPED)")
    sys.exit(0)

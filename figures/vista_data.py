"""
Pure data module for MVT-CS figures.
All values sourced from paper.tex tables; see comments for table references.
"""

# Budget grid (Table 1 / Table 3 x-axis)
BUDGETS = [256, 512, 1024, 2048, 4096, 8192]

# ---------------------------------------------------------------------------
# Table 1 (tab:budget-audit): M1 (Qwen3-8B)
# Columns: acc, brier, nll  per budget in BUDGETS order
# ---------------------------------------------------------------------------
M1_DATA = {
    "MMLU-Pro": {
        "acc":   [0.477, 0.526, 0.612, 0.696, 0.718, 0.738],
        # Note: §7.1 body gives 0.4917 for the best Brier (b=8192); table shows 0.492.
        # The 4-digit value 0.4917 is used where precision matters (see ORACLE_C).
        "brier": [0.832, 0.762, 0.637, 0.528, 0.508, 0.492],
        "nll":   [3.186, 2.885, 2.609, 2.458, 2.686, 2.763],
    },
    "ARC-Challenge": {
        "acc":   [0.920, 0.946, 0.951, 0.955, 0.956, 0.958],
        # Note: §7.1 body gives 0.0837 for best Brier (b=8192); table shows 0.084.
        "brier": [0.139, 0.098, 0.088, 0.088, 0.086, 0.084],
        "nll":   [0.466, 0.355, 0.371, 0.414, 0.462, 0.457],
    },
    "MedMCQA": {
        "acc":   [0.614, 0.624, 0.660, 0.662, 0.672, 0.666],
        "brier": [0.672, 0.667, 0.628, 0.646, 0.646, 0.662],
        "nll":   [2.297, 2.520, 2.856, 3.288, 3.649, 3.761],
    },
    "MedQA-USMLE": {
        "acc":   [0.648, 0.672, 0.718, 0.744, 0.776, 0.782],
        # Note: §7.1 body gives 0.4285 for best Brier (b=8192); table shows 0.429.
        "brier": [0.629, 0.585, 0.507, 0.462, 0.430, 0.429],
        "nll":   [2.361, 2.120, 1.934, 1.816, 2.053, 2.265],
    },
}

# ---------------------------------------------------------------------------
# Table 3 (tab:budget-audit-m2): M2 (DeepSeek-R1-Distill-Llama-8B)
# Columns: acc, brier only (no NLL column in paper)
# ---------------------------------------------------------------------------
M2_DATA = {
    "MMLU-Pro": {
        "acc":   [0.320, 0.382, 0.444, 0.496, 0.508, 0.502],
        "brier": [0.948, 0.937, 0.907, 0.886, 0.878, 0.888],
    },
    "ARC-Challenge": {
        "acc":   [0.775, 0.851, 0.875, 0.877, 0.872, 0.872],
        "brier": [0.366, 0.253, 0.237, 0.234, 0.238, 0.239],
    },
    "MedMCQA": {
        "acc":   [0.486, 0.518, 0.524, 0.518, 0.526, 0.526],
        "brier": [0.802, 0.851, 0.899, 0.929, 0.930, 0.929],
    },
    "MedQA-USMLE": {
        "acc":   [0.488, 0.506, 0.556, 0.570, 0.562, 0.568],
        "brier": [0.727, 0.740, 0.775, 0.805, 0.822, 0.827],
    },
}

# ---------------------------------------------------------------------------
# Table 5 (tab:gate0-oracle-c): Oracle C results
# best_b: budget with lowest Brier  (argmin over BUDGETS)
# S_star: Brier at best_b (4-digit where body text provides it)
# oracle_b: mean tokens used by Oracle C policy
# oracle_S: Brier achieved by Oracle C
# saving: compute saving fraction (0=none, 1=full)
# ---------------------------------------------------------------------------
ORACLE_C = {
    "M1": {
        "MMLU-Pro": {
            "best_b":   8192,
            "S_star":   0.4917,   # §7.1 body text: 4-digit; table shows 0.492
            "oracle_b": 461,
            "oracle_S": 0.4911,
            "saving":   0.944,
        },
        "ARC-Challenge": {
            "best_b":   8192,
            "S_star":   0.0837,   # §7.1 body text: 4-digit; table shows 0.084
            "oracle_b": 268,
            "oracle_S": 0.0834,
            "saving":   0.967,
        },
        "MedMCQA": {
            "best_b":   1024,
            "S_star":   0.6275,
            "oracle_b": 263,
            "oracle_S": 0.6248,
            "saving":   0.744,
        },
        "MedQA-USMLE": {
            "best_b":   8192,
            "S_star":   0.4285,   # §7.1 body text: 4-digit; table shows 0.429
            "oracle_b": 364,
            "oracle_S": 0.4274,
            "saving":   0.956,
        },
    },
    "M2": {
        "MMLU-Pro": {
            "best_b":   4096,
            "S_star":   0.8782,
            "oracle_b": 268,
            "oracle_S": 0.8776,
            "saving":   0.935,
        },
        "ARC-Challenge": {
            "best_b":   2048,
            "S_star":   0.2338,
            "oracle_b": 279,
            "oracle_S": 0.2333,
            "saving":   0.864,
        },
        "MedMCQA": {
            "best_b":   256,
            "S_star":   0.8019,
            "oracle_b": 256,
            "oracle_S": 0.8019,
            "saving":   0.000,
        },
        "MedQA-USMLE": {
            "best_b":   256,
            "S_star":   0.7268,
            "oracle_b": 256,
            "oracle_S": 0.7268,
            "saving":   0.000,
        },
    },
}

# ---------------------------------------------------------------------------
# Table 6 (tab:signals): Signal sign-AUC (MEAN only — no edge-by-edge data)
# sign_auc: mean across edges; spearman: mean Spearman rho with Brier change
# ---------------------------------------------------------------------------
SIGNALS = {
    "Confidence":          {"sign_auc": 0.757, "spearman": -0.082},
    "Entropy":             {"sign_auc": 0.758, "spearman": +0.086},
    "Top-1/top-2 margin":  {"sign_auc": 0.757, "spearman": -0.081},
    "Answer stability":    {"sign_auc": 0.545, "spearman": +0.314},
    "Confidence slope":    {"sign_auc": 0.488, "spearman": -0.028},
}

# ---------------------------------------------------------------------------
# Table 8 (tab:main): Policy comparison M1/DS1 (MMLU-Pro)
# ---------------------------------------------------------------------------
POLICY_COMPARISON = {
    "Best fixed (b=8192)":   {"tokens": 8192, "brier": 0.492, "nll": 2.763, "acc": 0.738, "net_saving": 0.000},
    "Confidence stopping":   {"tokens": 1453, "brier": 0.664, "nll": 2.952, "acc": 0.645, "net_saving": 0.823},
    "Entropy stopping":      {"tokens": 1624, "brier": 0.643, "nll": 2.938, "acc": 0.659, "net_saving": 0.802},
    "Stability stopping":    {"tokens": 1655, "brier": 0.608, "nll": 2.567, "acc": 0.644, "net_saving": 0.798},
    "VISTA (lambda=0)":      {"tokens": 8191, "brier": 0.492, "nll": 2.764, "acc": 0.738, "net_saving": 0.000, "ci": (-0.000, +0.000)},
    "VISTA (lambda=0.01)":   {"tokens": 7931, "brier": 0.498, "nll": 2.767, "acc": 0.734, "net_saving": 0.032, "ci": (0.027, 0.037)},
}

# ---------------------------------------------------------------------------
# Table 11 (tab:a07_rho_sensitivity): Audit-rate sensitivity (lambda=0)
# rho: audit rates; net_saving: list; brier: list; ci_lo, ci_hi: lists
# ---------------------------------------------------------------------------
RHO_SENSITIVITY = {
    "M1/MMLU-Pro": {
        "b_ref":      8192,
        "rho":        [0.05, 0.10, 0.20, 0.40],
        "net_saving": [+0.001, +0.001, +0.000, +0.000],
        "brier":      [0.4919, 0.4918, 0.4918, 0.4918],
        "ci_lo":      [0.000,  0.000, -0.000,  0.000],
        "ci_hi":      [0.002,  0.001,  0.000,  0.000],
    },
    "M1/ARC-Challenge": {
        "b_ref":      8192,
        "rho":        [0.05, 0.10, 0.20, 0.40],
        "net_saving": [+0.003, +0.001, +0.001, +0.000],
        "brier":      [0.0839, 0.0838, 0.0838, 0.0837],
        "ci_lo":      [0.001,  0.001,  0.000,  0.000],
        "ci_hi":      [0.005,  0.002,  0.001,  0.000],
    },
    "M1/MedMCQA": {
        "b_ref":      1024,
        "rho":        [0.05, 0.10, 0.20, 0.40],
        "net_saving": [-0.351, -0.687, -1.418, -2.786],
        "brier":      [0.6276, 0.6275, 0.6275, 0.6275],
        "ci_lo":      [-0.361, -0.700, -1.434, -2.807],
        "ci_hi":      [-0.341, -0.674, -1.402, -2.766],
    },
    "M2/MMLU-Pro": {
        "b_ref":      4096,
        "rho":        [0.05, 0.10, 0.20, 0.40],
        "net_saving": [-0.045, -0.099, -0.199, -0.398],
        "brier":      [0.8788, 0.8783, 0.8783, 0.8783],
        "ci_lo":      [-0.048, -0.100, -0.201, -0.400],
        "ci_hi":      [-0.041, -0.097, -0.197, -0.396],
    },
}

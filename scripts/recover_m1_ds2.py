#!/usr/bin/env python3
"""
Recovery script for 21 missing items in m1_ds2_4090 (ds2_0795 ~ ds2_0815).

Run on RTX 4090 (CUDA_VISIBLE_DEVICES=0):
  source /home/sclab/paper2/.venv/bin/activate
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=0 python recover_m1_ds2.py

Steps:
  1. gen-prefixes on 21-item subset → data/prefixes_m1_ds2_recovery_4090.jsonl
  2. run-d5 on recovery prefixes → data/results_m1_ds2_recovery_4090.json
  3. Merge recovery results into data/results_m1_ds2_4090.json
  4. Validate: exactly 1000 items, 6000 budget-level rows, 0 missing, 0 NaN.
"""

import json, subprocess, sys, time, shutil
from datetime import datetime
from pathlib import Path

VENV_PYTHON = "/home/sclab/paper2/.venv/bin/python3"
SCRIPT      = "/home/sclab/paper2/diag_logits.py"
BUDGETS     = "256,512,1024,2048,4096,8192"
UTIL        = "0.85"
MODEL       = "Qwen/Qwen3-8B"
GPU_TAG     = "4090"
THINK_END_ID = "151668"  # Qwen3 </think>

ITEMS_RECOVERY   = "data/items_ds2_recovery.json"
PREFIX_RECOVERY  = "data/prefixes_m1_ds2_recovery_4090.jsonl"
RESULT_RECOVERY  = "data/results_m1_ds2_recovery_4090.json"
RESULT_MAIN      = "data/results_m1_ds2_4090.json"
RESULT_BACKUP    = None   # set during run

ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_DEVICE_ORDER":    "PCI_BUS_ID",
    "PATH": "/home/sclab/miniconda3/envs/dualmoe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/home/sclab",
    "HF_HOME": "/home/sclab/.cache/huggingface",
}


def run(cmd, desc):
    print(f"\n[recover] {desc}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=ENV)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[recover] FAIL rc={proc.returncode} after {elapsed:.0f}s", flush=True)
        sys.exit(1)
    print(f"[recover] OK in {elapsed:.0f}s", flush=True)


def merge_and_validate():
    """Merge recovery results into main results file, validate."""
    main_d = json.load(open(RESULT_MAIN))
    rec_d  = json.load(open(RESULT_RECOVERY))

    main_results = main_d["results"]
    rec_results  = rec_d["results"] if isinstance(rec_d, dict) else rec_d

    # Check for duplicates
    existing_keys = {(r["question_id"], r["budget"]) for r in main_results}
    added = 0
    for r in rec_results:
        k = (r["question_id"], r["budget"])
        if k in existing_keys:
            print(f"[merge] SKIP duplicate: {k}", flush=True)
        else:
            main_results.append(r)
            existing_keys.add(k)
            added += 1

    print(f"[merge] Added {added} records (expected 21×6=126)", flush=True)

    # Validate
    qids   = set(r["question_id"] for r in main_results)
    budgets_seen = set(r["budget"]      for r in main_results)
    print(f"[validate] n_items={len(qids)}, total_records={len(main_results)}", flush=True)
    assert len(qids) == 1000, f"Expected 1000 items, got {len(qids)}"
    assert len(main_results) == 6000, f"Expected 6000 records, got {len(main_results)}"

    import math
    nan_count = sum(
        1 for r in main_results
        if any(math.isnan(p) or math.isinf(p) for p in (r.get("option_probs") or []))
    )
    assert nan_count == 0, f"NaN/Inf in option_probs: {nan_count}"

    # Backup original
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = RESULT_MAIN.replace(".json", f"_backup_{ts}.json")
    shutil.copy(RESULT_MAIN, backup)
    print(f"[merge] Backup saved: {backup}", flush=True)

    # Write merged
    main_d["n_results"] = len(main_results)
    main_d["results"]   = main_results
    json.dump(main_d, open(RESULT_MAIN, "w"), indent=2)
    print(f"[merge] Wrote {RESULT_MAIN} ({len(main_results)} records)", flush=True)
    print("[validate] ALL CHECKS PASSED", flush=True)


def main():
    print("=== M1/DS2 Recovery: 21 items (ds2_0795~ds2_0815) ===", flush=True)

    # Step 1: gen-prefixes
    if not Path(PREFIX_RECOVERY).exists():
        run([
            VENV_PYTHON, SCRIPT, "gen-prefixes",
            "--model",        MODEL,
            "--gpu-tag",      GPU_TAG,
            "--items",        ITEMS_RECOVERY,
            "--out",          PREFIX_RECOVERY,
            "--util",         UTIL,
            "--budgets",      BUDGETS,
            "--think-end-id", THINK_END_ID,
        ], f"gen-prefixes → {PREFIX_RECOVERY}")
    else:
        print(f"[recover] SKIP gen-prefixes — {PREFIX_RECOVERY} already exists", flush=True)

    # Step 2: run-d5
    if not Path(RESULT_RECOVERY).exists():
        run([
            VENV_PYTHON, SCRIPT, "run-d5",
            "--model",    MODEL,
            "--gpu-tag",  GPU_TAG,
            "--prefixes", PREFIX_RECOVERY,
            "--util",     UTIL,
            "--out",      RESULT_RECOVERY,
        ], f"run-d5 → {RESULT_RECOVERY}")
    else:
        print(f"[recover] SKIP run-d5 — {RESULT_RECOVERY} already exists", flush=True)

    # Step 3: merge + validate
    merge_and_validate()
    print("\n=== Recovery complete. m1_ds2_4090 now has 1000/1000 items. ===", flush=True)


if __name__ == "__main__":
    main()

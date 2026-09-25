#!/usr/bin/env python3
"""
Corpus generation coordinator — runs gen-prefixes + run-d5 for all cells.

GPU tracks (run in parallel):
  4090  (GPU 0): M1 Qwen3-8B,                DS1+DS2+DS3+DS4 (sequential)
  3090a (GPU 1): M2 DeepSeek-R1-Distill-Llama, DS1+DS2       (sequential)
  3090b (GPU 2): M2 DeepSeek-R1-Distill-Llama, DS3+DS4       (sequential)

Budgets: 256,512,1024,2048,4096,8192 (all 6 levels)
"""

import subprocess, sys, time, os
from pathlib import Path

VENV_PYTHON = "/home/sclab/paper2/.venv/bin/python3"
SCRIPT      = "/home/sclab/paper2/diag_logits.py"
BUDGETS     = "256,512,1024,2048,4096,8192"
UTIL        = "0.85"
LOG_DIR     = Path("/home/sclab/paper2/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set DRY_RUN=1 to run with 4-item _dry files and _dry output paths.
# All coordinator logic is identical; only file names change.
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

M1_MODEL = "Qwen/Qwen3-8B"
M2_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

M1_THINK_END = "151668"   # Qwen3 </think>
M2_THINK_END = "128014"   # DeepSeek-R1-Llama </think>

CELLS = [
    # (gpu_id, cuda_visible, gpu_tag, model, model_tag, think_end_id, ds_tag, items_file)
    # Track 0 — 4090: M1 all four datasets  [ACTIVE — Option A]
    (0, "0", "4090",  M1_MODEL, "m1", M1_THINK_END, "ds1", "data/items_ds1.json"),
    (0, "0", "4090",  M1_MODEL, "m1", M1_THINK_END, "ds2", "data/items_ds2.json"),
    (0, "0", "4090",  M1_MODEL, "m1", M1_THINK_END, "ds3", "data/items_ds3.json"),
    (0, "0", "4090",  M1_MODEL, "m1", M1_THINK_END, "ds4", "data/items_ds4.json"),
    # Track 1 — 3090a: M2 DS1+DS2
    (1, "1", "3090a", M2_MODEL, "m2", M2_THINK_END, "ds1", "data/items_ds1.json"),
    (1, "1", "3090a", M2_MODEL, "m2", M2_THINK_END, "ds2", "data/items_ds2.json"),
    # Track 2 — 3090b: M2 DS3+DS4
    (2, "2", "3090b", M2_MODEL, "m2", M2_THINK_END, "ds3", "data/items_ds3.json"),
    (2, "2", "3090b", M2_MODEL, "m2", M2_THINK_END, "ds4", "data/items_ds4.json"),
]

ENV_BASE = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "PATH": "/home/sclab/miniconda3/envs/dualmoe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/home/sclab",
    "HF_HOME": "/home/sclab/.cache/huggingface",
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
}


def run_cell(gpu_id, cuda_visible, gpu_tag, model, model_tag, think_end_id, ds_tag, items_file):
    """Run gen-prefixes then run-d5 for one (model, dataset, gpu) cell."""
    dry = "_dry" if DRY_RUN else ""
    if DRY_RUN:
        items_file = items_file.replace(".json", "_dry.json")
    prefix_out = f"data/prefixes_{model_tag}_{ds_tag}_{gpu_tag}{dry}.json"
    result_out  = f"data/results_{model_tag}_{ds_tag}_{gpu_tag}{dry}.json"

    env = {**ENV_BASE, "CUDA_VISIBLE_DEVICES": cuda_visible}

    tag = f"{model_tag}_{ds_tag}_{gpu_tag}"

    for step, (cmd_name, extra_args, out_file) in enumerate([
        ("gen-prefixes", [
            "--model",        model,
            "--gpu-tag",      gpu_tag,
            "--items",        items_file,
            "--out",          prefix_out,
            "--util",         UTIL,
            "--budgets",      BUDGETS,
            "--think-end-id", think_end_id,
        ], prefix_out),
        ("run-d5", [
            "--model",    model,
            "--gpu-tag",  gpu_tag,
            "--prefixes", prefix_out,
            "--util",     UTIL,
            "--out",      result_out,
        ], result_out),
    ]):
        if Path(out_file).exists():
            print(f"[{tag}] SKIP {cmd_name} — {out_file} already exists", flush=True)
            continue
        log_path = LOG_DIR / f"{tag}_{cmd_name.replace('-', '_')}.log"
        cmd = [VENV_PYTHON, SCRIPT, cmd_name] + extra_args

        print(f"[{tag}] START {cmd_name} → {out_file}", flush=True)
        t0 = time.time()

        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"[{tag}] FAIL {cmd_name} (rc={proc.returncode}) after {elapsed:.0f}s"
                  f" — see {log_path}", flush=True)
            return False
        print(f"[{tag}] DONE {cmd_name} in {elapsed:.0f}s", flush=True)

    return True


def run_track(track_cells):
    """Run a list of cells sequentially (same GPU)."""
    for cell in track_cells:
        ok = run_cell(*cell)
        if not ok:
            print(f"[track] ABORT remaining cells on this track due to failure", flush=True)
            break


def main():
    import threading

    # Group cells by track (gpu_id)
    tracks: dict[int, list] = {}
    for cell in CELLS:
        gpu_id = cell[0]
        tracks.setdefault(gpu_id, []).append(cell)

    print(f"Starting {len(tracks)} parallel GPU tracks", flush=True)
    for gid, cells in sorted(tracks.items()):
        ds_list = [c[6] for c in cells]
        model_tag = cells[0][4]
        gpu_tag   = cells[0][2]
        print(f"  Track {gid}: GPU {gpu_tag}, model={model_tag}, datasets={ds_list}", flush=True)

    threads = []
    t0_global = time.time()
    for gid in sorted(tracks):
        t = threading.Thread(target=run_track, args=(tracks[gid],), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed_total = time.time() - t0_global
    print(f"\nAll tracks finished in {elapsed_total/3600:.2f} h ({elapsed_total:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

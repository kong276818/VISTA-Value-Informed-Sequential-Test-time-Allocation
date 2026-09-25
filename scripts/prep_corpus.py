#!/usr/bin/env python3
"""
Prepare items JSON files for main corpus generation.

Outputs (written to data/):
  items_ds1.json  -- MMLU-Pro 1000 items (random seed=42)
  items_ds2.json  -- ARC-Challenge 1000 items (random seed=42)
  items_ds3.json  -- MedMCQA 500 items (GPQA-Diamond gated/unavailable; MedMCQA substituted)
  items_ds4.json  -- MedQA USMLE-4opt 500 items (random seed=42)

NOTE on DS3: GPQA-Diamond (Idavidrein/gpqa) is a gated HF dataset and was
not fully downloaded to the local cache. MedMCQA (medical MCQ, 4-choice,
validation split) is substituted. Both are difficult science/medicine MCQs
suitable for the MVT budget-sensitivity analysis.
"""

import json, random, sys
from pathlib import Path

SEED = 42
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DS1: MMLU-Pro — 1000 items, random sample seed=42 (14 subjects present)
# ---------------------------------------------------------------------------

def prep_mmlu_pro():
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    subjects = sorted(set(ds["category"]))
    print(f"  MMLU-Pro: {len(ds)} test items, {len(subjects)} subjects")

    all_rows = list(ds)
    rng = random.Random(SEED)
    selected = rng.sample(all_rows, 1000)

    items = []
    for i, row in enumerate(selected):
        ans_letter = row["answer"]
        ans_idx    = ord(ans_letter) - ord("A")
        items.append({
            "question_id":  f"ds1_{i:04d}",
            "source_id":    str(row.get("question_id", i)),
            "dataset":      "mmlu_pro",
            "category":     row["category"],
            "question":     row["question"],
            "options":      list(row["options"]),
            "answer":       ans_letter,
            "answer_index": ans_idx,
            "n_options":    len(row["options"]),
        })

    out = OUT_DIR / "items_ds1.json"
    out.write_text(json.dumps({"dataset": "mmlu_pro", "n": len(items), "items": items}, indent=2))
    print(f"  wrote {out}  ({len(items)} items)")


# ---------------------------------------------------------------------------
# DS2: ARC-Challenge — 1000 items
# ---------------------------------------------------------------------------

def prep_arc_challenge():
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print(f"  ARC-Challenge: {len(ds)} test items")

    all_rows = list(ds)
    rng = random.Random(SEED)
    selected = rng.sample(all_rows, min(1000, len(all_rows)))

    label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
                    "1": 0, "2": 1, "3": 2, "4": 3}
    items = []
    for i, row in enumerate(selected):
        labels = row["choices"]["label"]
        texts  = row["choices"]["text"]
        ans_key = row["answerKey"]
        if ans_key not in label_to_idx:
            continue
        ans_idx = label_to_idx[ans_key]
        sorted_labels = sorted(set(labels), key=lambda x: label_to_idx.get(x, 99))
        opts = [texts[labels.index(lbl)] for lbl in sorted_labels]
        ans_letter = chr(65 + ans_idx)
        items.append({
            "question_id":  f"ds2_{i:04d}",
            "source_id":    row["id"],
            "dataset":      "arc_challenge",
            "category":     "arc_challenge",
            "question":     row["question"],
            "options":      opts,
            "answer":       ans_letter,
            "answer_index": ans_idx,
            "n_options":    len(opts),
        })

    out = OUT_DIR / "items_ds2.json"
    out.write_text(json.dumps({"dataset": "arc_challenge", "n": len(items), "items": items}, indent=2))
    print(f"  wrote {out}  ({len(items)} items)")


# ---------------------------------------------------------------------------
# DS3: MedMCQA — 500 items (substituting GPQA-Diamond which is gated)
# ---------------------------------------------------------------------------

def prep_medmcqa():
    from datasets import load_dataset
    ds = load_dataset("medmcqa", split="validation", trust_remote_code=True)
    # Keep only single-choice items (cop is unambiguous)
    single = [row for row in ds if row.get("choice_type", "single") == "single"]
    print(f"  MedMCQA validation: {len(ds)} total, {len(single)} single-choice")

    rng = random.Random(SEED)
    selected = rng.sample(single, min(500, len(single)))

    items = []
    for i, row in enumerate(selected):
        opts = [row["opa"], row["opb"], row["opc"], row["opd"]]
        ans_idx = int(row["cop"])    # 0-based: 0=A, 1=B, 2=C, 3=D
        ans_letter = chr(65 + ans_idx)
        items.append({
            "question_id":  f"ds3_{i:04d}",
            "source_id":    str(row.get("id", i)),
            "dataset":      "medmcqa",
            "category":     row.get("subject_name", "medmcqa"),
            "question":     row["question"],
            "options":      opts,
            "answer":       ans_letter,
            "answer_index": ans_idx,
            "n_options":    4,
        })

    out = OUT_DIR / "items_ds3.json"
    out.write_text(json.dumps({
        "dataset": "medmcqa",
        "note": "GPQA-Diamond gated/unavailable; MedMCQA (validation, single-choice) substituted",
        "n": len(items),
        "items": items,
    }, indent=2))
    print(f"  wrote {out}  ({len(items)} items)")


# ---------------------------------------------------------------------------
# DS4: MedQA USMLE 4-option — 500 items
# ---------------------------------------------------------------------------

def prep_medqa():
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    print(f"  MedQA (GBaker 4-opt): {len(ds)} test items")

    all_rows = list(ds)
    rng = random.Random(SEED)
    selected = rng.sample(all_rows, min(500, len(all_rows)))

    # answer_idx is a letter string "A"/"B"/"C"/"D"
    items = []
    for i, row in enumerate(selected):
        opts_dict = row["options"]                   # dict {"A": ..., "B": ..., ...}
        opts = [opts_dict[k] for k in sorted(opts_dict.keys())]
        ans_key = str(row["answer_idx"])             # letter "A"/"B"/"C"/"D"
        if ans_key.isdigit():
            ans_idx = int(ans_key) - 1
        else:
            ans_idx = ord(ans_key.upper()) - ord("A")
        ans_letter = chr(65 + ans_idx)
        items.append({
            "question_id":  f"ds4_{i:04d}",
            "source_id":    str(row.get("id", i)),
            "dataset":      "medqa_usmle",
            "category":     "medqa_usmle",
            "question":     row["question"],
            "options":      opts,
            "answer":       ans_letter,
            "answer_index": ans_idx,
            "n_options":    len(opts),
        })

    out = OUT_DIR / "items_ds4.json"
    out.write_text(json.dumps({"dataset": "medqa_usmle", "n": len(items), "items": items}, indent=2))
    print(f"  wrote {out}  ({len(items)} items)")


if __name__ == "__main__":
    print("=== DS1: MMLU-Pro ===")
    prep_mmlu_pro()
    print("\n=== DS2: ARC-Challenge ===")
    prep_arc_challenge()
    print("\n=== DS3: MedMCQA (GPQA substituted) ===")
    prep_medmcqa()
    print("\n=== DS4: MedQA USMLE ===")
    prep_medqa()
    print("\nAll datasets prepared.")

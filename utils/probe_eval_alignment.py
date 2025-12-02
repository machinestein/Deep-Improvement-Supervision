#!/usr/bin/env python3
import os, sys, json, glob, argparse
from collections import Counter, defaultdict
import numpy as np

def base_name(name: str) -> str:
    # Strip augmentation suffixes like "|||t0|||..."
    if "|||" in name:
        return name.split("|||", 1)[0]
    return name

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_identifiers(root):
    p = os.path.join(root, "identifiers.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"identifiers.json not found under {root}")
    obj = load_json(p)
    if isinstance(obj, list):
        id2name = {i: v for i, v in enumerate(obj)}
    elif isinstance(obj, dict):
        id2name = {int(k): v for k, v in obj.items()}
    else:
        raise ValueError("Unsupported identifiers.json format")
    name2id = {}
    for k, v in id2name.items():
        if v not in name2id:
            name2id[v] = k
    return id2name, name2id

def collect_npys(root, split, field="puzzle_identifiers"):
    files = sorted(glob.glob(os.path.join(root, split, f"*__{field}.npy")))
    arrays = []
    for fp in files:
        try:
            a = np.load(fp, mmap_mode="r")
            arrays.append(np.array(a).reshape(-1))
        except Exception as e:
            print(f"Warning: failed to load {fp}: {e}")
    if arrays:
        return np.concatenate(arrays), files
    return np.array([], dtype=np.int64), files

def main():
    ap = argparse.ArgumentParser(description="Probe ARC identifier alignment & augmentation naming across train/test roots.")
    ap.add_argument("train_root")
    ap.add_argument("test_root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--show", type=int, default=10, help="Show up to N sample rows for diagnostics.")
    args = ap.parse_args()

    train_id2name, train_name2id = load_identifiers(args.train_root)
    test_id2name,  test_name2id  = load_identifiers(args.test_root)

    # Canonical (base) name views
    train_base = {k: base_name(v) for k, v in train_id2name.items()}
    test_base  = {k: base_name(v) for k, v in test_id2name.items()}

    train_names = set(train_name2id.keys())
    test_names  = set(test_name2id.keys())
    train_base_names = set(train_base.values())
    test_base_names  = set(test_base.values())

    shared_names = train_names & test_names
    shared_base  = train_base_names & test_base_names

    print("=== Identifier-space overview ===")
    print(f"Train root: {args.train_root}")
    print(f"Test  root: {args.test_root}")
    print(f"Train id count: {len(train_id2name)}")
    print(f"Test  id count: {len(test_id2name)}")
    print(f"Shared puzzle names (literal): {len(shared_names)}")
    # literal mismatches by numeric ids
    if shared_names:
        mism = sum(1 for n in shared_names if train_name2id[n] != test_name2id[n])
        print(f"Shared names with different numeric ids: {mism}")
    print(f"Shared base puzzle names (after stripping '|||…'): {len(shared_base)}")
    only_in_train_base = sorted(list(train_base_names - test_base_names))
    only_in_test_base  = sorted(list(test_base_names - train_base_names))
    print(f"Only-in-train base names: {len(only_in_train_base)}")
    print(f"Only-in-test  base names: {len(only_in_test_base)}")

    # Evaluate clamping risk (ids outside train range) for actual test files
    test_ids, files = collect_npys(args.test_root, args.split, "puzzle_identifiers")
    if test_ids.size:
        train_max = max(train_id2name.keys())
        frac_clamped = float((test_ids >= (train_max + 1)).mean())
        print("\n=== Eval clamping risk (based on test files) ===")
        print(f"Files scanned: {len(files)}")
        print(f"Fraction of test ids that would be clamped (>= train_id_count): {frac_clamped:.3f}")
        # Also measure how many test ids map to a train base-name (join via base name)
        test_names_from_ids = [test_id2name.get(int(i), None) for i in test_ids[:100000]]  # sample up to 100k
        test_bases = [base_name(n) if n is not None else None for n in test_names_from_ids]
        hits = sum(1 for b in test_bases if b is not None and b in train_base_names)
        print(f"Among first {min(100000, test_ids.size)} test examples: "
              f"{hits} have base puzzle present in TRAIN ({hits / max(1, min(100000, test_ids.size)):.3f}).")
    else:
        print("\nNo '*__puzzle_identifiers.npy' found under test split; skipping clamping check.")

    # Try to find canonical join conflicts: test name → base name that isn't in test_puzzles.json
    tpp = os.path.join(args.test_root, "test_puzzles.json")
    if os.path.exists(tpp):
        test_puzzles = load_json(tpp)  # keys should be canonical base puzzle ids
        canonical = set(test_puzzles.keys())
        print("\n=== Canonical name alignment vs test_puzzles.json ===")
        print(f"Canonical (test_puzzles.json) count: {len(canonical)}")
        in_test_but_not_canonical = sorted(list(test_base_names - canonical))
        in_canonical_but_missing  = sorted(list(canonical - test_base_names))
        print(f"Names seen in TEST identifiers (base) but not in test_puzzles.json: {len(in_test_but_not_canonical)}")
        print(f"Names in test_puzzles.json but not present in TEST identifiers (base): {len(in_canonical_but_missing)}")

        if in_test_but_not_canonical and args.show > 0:
            print("\nExamples present in TEST identifiers (base) but not in test_puzzles.json:")
            for n in in_test_but_not_canonical[:args.show]:
                print(" ", n)

        if in_canonical_but_missing and args.show > 0:
            print("\nExamples present in test_puzzles.json but missing from TEST identifiers (base):")
            for n in in_canonical_but_missing[:args.show]:
                print(" ", n)
    else:
        print("\nNo test_puzzles.json under test root; cannot check canonical join.")

    # Build a proposed remap: test id -> train id via base-name join (only where possible)
    # This is useful if you want to reindex test ids onto train's id space.
    base_to_train_id = {}
    for tid, name in train_id2name.items():
        base_to_train_id[base_name(name)] = tid
    remap = {}
    misses = 0
    for tid, name in test_id2name.items():
        b = base_name(name)
        if b in base_to_train_id:
            remap[tid] = base_to_train_id[b]
        else:
            misses += 1

    print("\n=== Proposed test→train id remap (via base-name) ===")
    print(f"Coverage: {len(remap)}/{len(test_id2name)} ids ({len(remap)/max(1,len(test_id2name)):.3f})")
    print(f"Unmapped (no base-name match in TRAIN): {misses}")

    # Show a few concrete mismatches (same base name but different numeric ids)
    diffs = []
    for name in (train_names & test_names):
        if train_name2id[name] != test_name2id[name]:
            diffs.append((name, train_name2id[name], test_name2id[name]))
    if diffs and args.show > 0:
        print("\nSample literal-name numeric mismatches (name, train_id, test_id):")
        for row in diffs[:args.show]:
            print(" ", row)

    print("\nDone.")

if __name__ == "__main__":
    main()

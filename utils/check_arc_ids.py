#!/usr/bin/env python3
import os, sys, json, glob
import numpy as np
from collections import Counter

def load_id_map(root):
    """Return {int_id: puzzle_name} from root/identifiers.json."""
    p = os.path.join(root, "identifiers.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"identifiers.json not found under {root}")
    with open(p, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return {i: v for i, v in enumerate(obj)}
    if isinstance(obj, dict):
        # keys may be strings in JSON
        return {int(k): v for k, v in obj.items()}
    raise ValueError("Unsupported identifiers.json format")

def invert(d):
    """{id:name} -> {name:id} (warn if duplicates)."""
    inv = {}
    for k, v in d.items():
        if v in inv and inv[v] != k:
            # same name with multiple ids (shouldn't happen if consistent)
            pass
        inv[v] = k
    return inv

def collect_split_ids(root, split="test"):
    """Concatenate all ids from *__puzzle_identifiers.npy under root/<split>/."""
    split_dir = os.path.join(root, split)
    files = sorted(glob.glob(os.path.join(split_dir, "*__puzzle_identifiers.npy")))
    if not files:
        return np.array([], dtype=np.int64), []
    arrays = []
    for fp in files:
        try:
            a = np.load(fp)
            arrays.append(a.reshape(-1))
        except Exception as e:
            print(f"Warning: failed to load {fp}: {e}")
    if arrays:
        return np.concatenate(arrays), files
    return np.array([], dtype=np.int64), files

def main(train_root, test_root, split="test"):
    train_idmap = load_id_map(train_root)
    test_idmap  = load_id_map(test_root)

    train_name2id = invert(train_idmap)
    test_name2id  = invert(test_idmap)

    train_names = set(train_name2id.keys())
    test_names  = set(test_name2id.keys())

    shared      = sorted(train_names & test_names)
    only_train  = sorted(train_names - test_names)
    only_test   = sorted(test_names - train_names)

    print("=== Identifier-space overview ===")
    print(f"Train root: {train_root}")
    print(f"Test  root: {test_root}")
    print(f"Train id count: {len(train_idmap)}")
    print(f"Test  id count: {len(test_idmap)}")
    print(f"Shared puzzle names: {len(shared)}")
    print(f"Only-in-train names: {len(only_train)}")
    print(f"Only-in-test  names: {len(only_test)}")

    if shared:
        mismatched = [(n, train_name2id[n], test_name2id[n])
                      for n in shared if train_name2id[n] != test_name2id[n]]
        print(f"Shared names with different numeric ids: {len(mismatched)}")
        if mismatched:
            print("  Example mismatches (name, train_id, test_id):")
            for row in mismatched[:10]:
                print("   ", row)

    # Clamping risk analysis (what fraction of test ids would be >= train_max?)
    test_ids, files = collect_split_ids(test_root, split=split)
    if test_ids.size == 0:
        print(f"No '*__puzzle_identifiers.npy' found under {os.path.join(test_root, split)}.")
        print("Cannot estimate clamping risk from files; relying only on identifiers.json.")
    else:
        train_max = max(train_idmap.keys()) if train_idmap else -1
        frac_clamped = float((test_ids >= (train_max + 1)).mean())
        unique_clamped = int(np.unique(test_ids[test_ids >= (train_max + 1)]).size)
        print("\n=== Eval clamping risk (based on actual test files) ===")
        print(f"Files scanned: {len(files)}")
        print(f"Fraction of test ids that would be clamped (>= train_id_count): {frac_clamped:.3f}")
        print(f"Unique test ids outside train range: {unique_clamped}")

    # Show a few examples present in test but absent in train
    if only_test:
        print("\nExamples of puzzles present only in TEST mapping:")
        for n in only_test[:10]:
            print(" ", n, "→ test_id", test_name2id[n])

if __name__ == "__main__":
    #if len(sys.argv) != 3:
    #    print("Usage: python check_arc_ids.py <TRAIN_DATA_ROOT> <TEST_DATA_ROOT>")
    #    sys.exit(1)
    main('data/arc_with_steps_new_match', 'data/arc1concept-aug-1000')

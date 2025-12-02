#!/usr/bin/env python3
import os, json, argparse, glob
import numpy as np

def canon(name: str) -> str:
    return name.split("|||", 1)[0]

def load_idlist(root):
    with open(os.path.join(root, "identifiers.json"), "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    # dict fallback (not expected for the gold builder)
    out = []
    for k in range(max(map(int, obj.keys())) + 1):
        out.append(obj.get(str(k), "<blank>"))
    return out

def load_npys(root, split, field):
    files = sorted(glob.glob(os.path.join(root, split, f"*__{field}.npy")))
    arrs = [np.load(fp, mmap_mode="r") for fp in files]
    return np.concatenate([a.reshape(-1) for a in arrs]) if arrs else np.array([], dtype=np.int64)

def main(train_root, eval_root):
    train_ids = load_idlist(train_root)
    eval_ids  = load_idlist(eval_root)

    # 1) Name->id maps
    t_map = {name: i for i, name in enumerate(train_ids) if i > 0 and name != "<blank>"}
    e_map = {name: i for i, name in enumerate(eval_ids)  if i > 0 and name != "<blank>"}

    # 2) Overlap on names (canonical)
    t_canon = {canon(n) for n in t_map.keys()}
    e_canon = {canon(n) for n in e_map.keys()}
    shared_canon = t_canon & e_canon
    print(f"Shared canonical puzzle names: {len(shared_canon)}")
    print(f"Only in TRAIN (canon): {len(t_canon - e_canon)}")
    print(f"Only in EVAL  (canon): {len(e_canon - t_canon)}")

    # 3) Check that overlapping names map to SAME ids (after alignment)
    same_id = 0
    checked = 0
    for name in set(t_map.keys()) & set(e_map.keys()):
        if t_map[name] == e_map[name]:
            same_id += 1
        checked += 1
    if checked:
        print(f"Literal-name overlap with identical ids: {same_id}/{checked} ({same_id/checked:.3f})")

    # 4) Sample actual rows from TRAIN and EVAL splits, map ids->canon name, verify presence
    t_pids = load_npys(train_root, "train", "puzzle_identifiers")
    e_pids = load_npys(eval_root,  "test",  "puzzle_identifiers")
    def id2canon(idlist, pid):
        if pid < 0 or pid >= len(idlist): return None
        name = idlist[int(pid)]
        return None if name in (None, "<blank>") else canon(name)

    if t_pids.size and e_pids.size:
        t_names = set(filter(None, (id2canon(train_ids, pid) for pid in t_pids[:100000])))
        e_names = set(filter(None, (id2canon(eval_ids,  pid) for pid in e_pids[:100000])))
        inter = len(t_names & e_names)
        print(f"Sampled canonical-name overlap (100k rows): {inter} / TRAIN:{len(t_names)} / EVAL:{len(e_names)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--eval_root",  required=True)
    args = ap.parse_args()
    main(args.train_root, args.eval_root)

# python3 verify_alignment.py \
#   --train_root data/arc_with_steps_new_match \
#   --eval_root  data/arc1concept-aug-1000
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np

# ===== Token/Vocabulary =====
PAD_ID = 0
EOS_ID = 1
# ARC colors are 0..9. We encode them as +2 so that:
#   0..9  ->  2..11
# making vocab_size = 12
VOCAB_SIZE = 12

# ===== Geometry / Canvas =====
ARC_MAX_GRID_SIZE = 30
SEQ_LEN = ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE  # 900

# ===== Dihedral Group (D4) =====
# mapping: id -> transform
# 0: identity
# 1: rot90 (CCW)
# 2: rot180
# 3: rot270
# 4: flip left-right
# 5: flip up-down
# 6: transpose (main diagonal)
# 7: anti-diagonal reflection (flip LR after rot90)
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    tid = int(tid) % 8
    if tid == 0:
        return arr
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)
    elif tid == 5:
        return np.flipud(arr)
    elif tid == 6:
        return arr.T
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    else:
        return arr

def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[int(tid) % 8])

# ===== Colors / Permutations =====
def random_color_permutation(rng: np.random.Generator) -> np.ndarray:
    """Return a permutation array p of length 10 so that new_color = p[old_color].
    We keep 0 fixed (background), and randomly permute 1..9.
    """
    p = np.arange(10, dtype=np.uint8)
    tail = p[1:].copy()
    rng.shuffle(tail)
    p[1:] = tail
    return p

def apply_color_permutation(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    #assert perm.shape == (10,), "perm must have length 10 (colors 0..9)"
    arr = np.clip(arr.astype(np.int16), 0, 9).astype(np.uint8)
    return perm[arr]

# ===== Encoding to tokens =====
def _place_grid_on_canvas(grid: np.ndarray, top: int, left: int, canvas: np.ndarray,
                           eval_compat_sequences: bool = False) -> Tuple[int, int, int, int]:
    """Place grid values (+2) at (top,left) on canvas (which is already filled with PAD=0).
    Also writes an EOS row and EOS col (token=1) right after the grid region, if within bounds.
    Returns (y0,x0,h,w).
    
    If eval_compat_sequences=True, it omits the corner (y0+h, x0+w) pixel
    to match the build_arc_dataset.py tokenization.
    """
    h, w = grid.shape
    y0, x0 = int(top), int(left)
    # map colors to token-ids (+2); zeros within grid are valid foreground value -> 2
    canvas[y0:y0+h, x0:x0+w] = (grid.astype(np.uint8) + 2)
    
    # EOS row
    if y0 + h < canvas.shape[0]:
        canvas[y0 + h, x0:x0+w] = EOS_ID
        
    # EOS col
    if x0 + w < canvas.shape[1]:
        canvas[y0:y0+h, x0 + w] = EOS_ID
        
    # ===== THIS IS THE FIX =====
    # The eval script (build_arc_dataset.py) does NOT write the corner pixel.
    # The original train script *did* write it.
    # We now only write the corner if NOT in eval compatibility mode.
    if not eval_compat_sequences:
        if (y0 + h < canvas.shape[0]) and (x0 + w < canvas.shape[1]):
            canvas[y0 + h, x0 + w] = EOS_ID
    # ===========================
            
    return y0, x0, h, w

def _choose_translation_offset(frames: List[np.ndarray], canvas_size: int, rng: Optional[np.random.Generator], do_translation: bool) -> Tuple[int,int]:
    """Choose a single (top,left) so that all frames fit on the canvas; if do_translation, choose uniformly, else (0,0)."""
    max_h = max(int(f.shape[0]) for f in frames)
    max_w = max(int(f.shape[1]) for f in frames)
    H = W = int(canvas_size)
    assert max_h <= H and max_w <= W, f"Frame too big ({max_h},{max_w}) for canvas {H}."
    max_top = H - max_h
    max_left = W - max_w
    if do_translation and (rng is not None):
        top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
        left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    else:
        top = 0
        left = 0
    return top, left

    # Put these near the other helpers
def sanitize_colors(arr: np.ndarray, lo: int = 0, hi: int = 9) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int16)  # avoid uint8 wrap
    arr = np.clip(arr, lo, hi)
    return arr.astype(np.uint8)

def clip_to_canvas(arr: np.ndarray, max_size: int) -> np.ndarray:
    """Top-left crop to fit within max_size x max_size."""
    arr = np.asarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    h2 = min(h, max_size)
    w2 = min(w, max_size)
    return arr[:h2, :w2]


def grids_to_sequences(frames: List[np.ndarray], canvas_size: int = ARC_MAX_GRID_SIZE,
                       rng: Optional[np.random.Generator] = None, do_translation: bool = True,
                       eval_compat_sequences: bool = False) -> List[np.ndarray]: # <--- ADDED FLAG
    """Encode a list of 2D grids into token sequences (flattened) with shared translation offset."""
    # NEW: sanitize & clip every frame first
    frames = [clip_to_canvas(sanitize_colors(np.asarray(g, dtype=np.uint8)), canvas_size)
              for g in frames]

    top, left = _choose_translation_offset(frames, canvas_size, rng, do_translation)
    seqs: List[np.ndarray] = []
    for grid in frames:
        H = W = int(canvas_size)
        canvas = np.zeros((H, W), dtype=np.uint8)  # PAD=0
        # Pass the flag down to the helper
        _place_grid_on_canvas(grid, top, left, canvas, eval_compat_sequences=eval_compat_sequences)
        seqs.append(canvas.reshape(-1))
    return seqs


# def grids_to_sequences(frames: List[np.ndarray], canvas_size: int = ARC_MAX_GRID_SIZE,
#                        rng: Optional[np.random.Generator] = None, do_translation: bool = True) -> List[np.ndarray]:
#     """Encode a list of 2D grids into token sequences (flattened) with shared translation offset.
#     Returns a list of 1D np.uint8 arrays, each length canvas_size*canvas_size."""
#     top, left = _choose_translation_offset(frames, canvas_size, rng, do_translation)
#     seqs: List[np.ndarray] = []
#     for grid in frames:
#         H = W = int(canvas_size)
#         canvas = np.zeros((H, W), dtype=np.uint8)  # filled with PAD=0
#         _place_grid_on_canvas(grid, top, left, canvas)
#         seqs.append(canvas.reshape(-1))
#     return seqs

# ===== Trajectory helpers =====
def normalize_steps(input_grid: np.ndarray,
                    steps: Optional[List[np.ndarray]],
                    output_grid: np.ndarray,
                    target_num_steps: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return exactly target_num_steps intermediate frames and a mask (1 if the step is real, 0 if padded).
    If steps is None or shorter than target_num_steps, we pad by repeating the last available frame
    (preferring provided steps; fall back to output; and finally input)."""
    mask = np.zeros((target_num_steps,), dtype=np.uint8)
    frames: List[np.ndarray] = []
    if steps is None:
        steps = []
    steps_np: List[np.ndarray] = []
    for s in steps:
        steps_np.append(np.array(s, dtype=np.uint8))
    for i in range(min(target_num_steps, len(steps_np))):
        frames.append(steps_np[i])
        mask[i] = 1
    last = steps_np[-1] if len(steps_np) > 0 else output_grid if output_grid is not None else input_grid
    while len(frames) < target_num_steps:
        frames.append(last)
        # mask stays 0 for padded slots
    return frames, mask

@dataclass
class AugSpec:
    tid: int               # dihedral transform id 0..7
    color_perm: np.ndarray # permutation array of shape (10,)

def augment_frames(frames: List[np.ndarray], spec: AugSpec) -> List[np.ndarray]:
    """Apply the same augmentation (dihedral + color permutation) to each frame."""
    out: List[np.ndarray] = []
    for fr in frames:
        fr2 = dihedral_transform(fr, spec.tid)
        fr3 = apply_color_permutation(fr2, spec.color_perm)
        out.append(fr3.astype(np.uint8))
    return out

def trajectory_hash(frames: List[np.ndarray]) -> str:
    """Content hash for deduplication across all frames in one puzzle (transition-level)."""
    h = 0
    for fr in frames:
        arr = np.ascontiguousarray(fr, dtype=np.uint8)
        h = (h * 1315423911) ^ int(arr.sum()) ^ int(arr.size) ^ int(arr.max()) ^ int(arr.min())
    return hex(h & 0xFFFFFFFFFFFFFFFF)[2:]
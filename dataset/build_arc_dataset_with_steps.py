from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata, dihedral_transform, inverse_dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file_prefix: str
    output_dir: str
    subsets: List[str]
    test_set_name: str
    test_set_name2: str = "your_test_set"
    seed: int = 42
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1 # start > 1 to handle multiple datasets
    
ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5
TARGET_STEP_COUNT = 10 # Added constant for step count

PuzzleIdSeparator = "|||"
    

@dataclass
class ARCPuzzle:
    id: str
    # Examples now contain a dict with input, output, and intermediate steps
    examples: List[Dict[str, Any]] # MODIFIED: from np.ndarray to Any

    
def arc_grid_to_np(grid: List[List[int]]):
    # try: # MODIFIED: Removed try/except block
    arr = np.array(grid) # Let it raise ValueError on ragged arrays
    # except ValueError:
    #     ... (removed) ...

    # Handle `[]` input, which creates a 1D array `array([])`
    if arr.ndim == 1 and arr.shape[0] == 0:
        arr = np.array([], dtype=np.uint8).reshape(0, 0)

    # Shape check
    assert arr.ndim == 2, f"Grid did not convert to 2D array. Original: {grid}"
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # --- MODIFICATION ---
    # Element check & clamp instead of assert
    arr = np.clip(arr, 0, 9)
    # --- END MODIFICATION ---
    return arr.astype(np.uint8)


def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, steps: List[np.ndarray], do_translation: bool):
    """
    Applies the same translational augmentation (padding) to input, output, and all intermediate steps.
    
    Args:
        inp: Input grid (H, W)
        out: Output grid (H, W)
        steps: List of intermediate step grids [ (H, W), (H', W'), ... ]
        do_translation: Whether to apply random padding
        
    Returns:
        A tuple of (inp_flat, out_flat, steps_flat_stacked)
        inp_flat: (L,)
        out_flat: (L,)
        steps_flat_stacked: (N_steps, L)
    """
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    
    # Determine max shape across all grids (input, output, and all steps)
    all_grids = [inp, out] + steps # MODIFIED: Use the list directly
    
    # Filter out any (0,0) grids from shape calculation
    valid_grids = [g for g in all_grids if g.ndim == 2 and g.size > 0]
    if not valid_grids:
         # All grids are empty, just use 0
        max_shape_0 = 0
        max_shape_1 = 0
    else:
        max_shape_0 = max(g.shape[0] for g in valid_grids)
        max_shape_1 = max(g.shape[1] for g in valid_grids)

    # Compute random top-left pad
    if do_translation:
        pad_r = np.random.randint(0, ARCMaxGridSize - max_shape_0 + 1)
        pad_c = np.random.randint(0, ARCMaxGridSize - max_shape_1 + 1)
    else:
        pad_r = pad_c = 0

    processed_grids = []
    for grid in all_grids:
        nrow, ncol = grid.shape
        padded_grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)), constant_values=0)

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            padded_grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            padded_grid[pad_r:eos_row, eos_col] = 1
            
        processed_grids.append(padded_grid.flatten())

    # Unpack the results
    inp_flat = processed_grids[0]
    out_flat = processed_grids[1]
    steps_flat = processed_grids[2:]
    
    # Stack the steps
    steps_flat_stacked = np.stack(steps_flat, axis=0)
    
    return inp_flat, out_flat, steps_flat_stacked


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    # Hash the puzzle for checking equivalence
    hashes = []
    for example_type, example in puzzle.items():
        for ex_data in example.examples:
            hashes.append(f"{grid_hash(ex_data['input'])}|{grid_hash(ex_data['output'])}")
            
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def aug(name: str):
    # Augment plan
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))])  # Permute colors, Excluding "0" (black)
    
    name_with_aug_repr = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)
    
    return name_with_aug_repr, _map_grid


def inverse_aug(name: str):
    # Inverse the "aug" function
    if PuzzleIdSeparator not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # Remove "t" letter
    inv_perm = np.argsort(list(perm)).astype(np.uint8)
    
    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]
    
    return name.split(PuzzleIdSeparator)[0], _map_grid


def convert_single_arc_puzzle(results: dict, name: str, puzzle: dict, aug_count: int, dest_mapping: Dict[str, Tuple[str, str]]):
    # Convert
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}
    
    for example_type, examples in puzzle.items():
        # Map to target split
        dest = dest_mapping[example_type]
        
        for example in examples:
            try:
                inp = arc_grid_to_np(example["input"])
                out = arc_grid_to_np(example["output"])
            except (ValueError, AssertionError) as e:
                print(f"FATAL: Failed to parse input/output for puzzle {name}. Skipping example. Error: {e}")
                continue # Skip this example
            
            # --- MODIFICATION ---
            # Process the "steps" field and ensure it has TARGET_STEP_COUNT
            steps_data = []
            if "steps" in example:
                # Load existing steps
                for step_grid in example["steps"]:
                    try:
                        steps_data.append(arc_grid_to_np(step_grid))
                    except (ValueError, AssertionError) as e:
                        # Catch ragged arrays (ValueError) or shape/size asserts
                        print(f"Warning: Malformed step grid in {name}. Content: {step_grid}. Replacing with output grid. Error: {e}")
                        steps_data.append(out) # Use 'out' grid as fallback
                
                num_existing_steps = len(steps_data)
                if num_existing_steps < TARGET_STEP_COUNT:
                    # Pad with output grid
                    padding_needed = TARGET_STEP_COUNT - num_existing_steps
                    steps_data.extend([out] * padding_needed)
                elif num_existing_steps > TARGET_STEP_COUNT:
                    # Truncate to target count
                    steps_data = steps_data[:TARGET_STEP_COUNT]
            else:
                # If no steps, create full path by repeating output
                steps_data = [out] * TARGET_STEP_COUNT
                
            # steps_np = np.stack(steps_data, axis=0) # <-- MODIFIED: REMOVED THIS LINE
            # --- END MODIFICATION ---

            converted[dest].examples.append({
                "input": inp,
                "output": out,
                "steps": steps_data # <-- MODIFIED: Store the list, not the stacked array
            })

    group = [converted]
    
    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, _map_grid = aug(name)

            # Check duplicate
            augmented = {}
            for dest, puzzle_data in converted.items():
                aug_examples = []
                for ex in puzzle_data.examples:
                    aug_examples.append({
                        "input": _map_grid(ex["input"]),
                        "output": _map_grid(ex["output"]),
                        "steps": [_map_grid(s) for s in ex["steps"]] # MODIFIED: Augment list of steps
                    })
                augmented[dest] = ARCPuzzle(aug_name, aug_examples)

            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)
                
            if len(group) >= aug_count + 1:
                break
            
        if len(group) < aug_count + 1:
            print (f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append
    for dest in dests:
        # Convert the examples
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_puzzles_arcagi(config: DataProcessConfig):
    train_examples_dest = ("train", "all")
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        config.test_set_name2: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))]
    }
    
    test_puzzles = {}
    results = {}

    total_puzzles = 0
    for subset_name in config.subsets:
        # --- MODIFICATION ---
        # Handle the new training file name
        if subset_name == "training":
            # Use the exact filename provided by the user for challenges
            challenges_filename = f"{config.input_file_prefix}_training_converted_k10.json"
            print(f"Loading special training subset from: {challenges_filename}")
        else:
            # Use the original pattern for other subsets (like test sets)
            challenges_filename = f"{config.input_file_prefix}_{subset_name}_challenges.json"
            print(f"Loading standard subset: {challenges_filename}")
        
        # ALWAYS look for the solutions file using the standard pattern
        sols_filename = f"{config.input_file_prefix}_{subset_name}_solutions.json"
        # --- END MODIFICATION ---

        # Load all puzzles in this subset
        with open(challenges_filename, "r") as f:
            puzzles = json.load(f)

        # --- MODIFICATION ---
        # Only load solutions if sols_filename is set and the file exists
        if os.path.isfile(sols_filename):
        # --- END MODIFICATION ---
            with open(sols_filename, "r") as f:
                sols = json.load(f)
                
                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            # Fill with dummy
            print (f"{subset_name} solutions file not found ('{sols_filename}'), filling with dummy")

            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)
        
        # Assign by fraction
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(subset_name, test_examples_map["_default"]):
                if fraction < f:
                    test_examples_dest = dest
                    break
                    
            assert test_examples_dest is not None
            
            if test_examples_dest[0] == "test":
                test_puzzles[name] = puzzle
                
            convert_single_arc_puzzle(results, name, puzzle, config.num_aug, {"train": train_examples_dest, "test": test_examples_dest})
            total_puzzles += 1

    print (f"Total puzzles: {total_puzzles}")
    return results, test_puzzles


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)
    
    # Read dataset
    data, test_puzzles = load_puzzles_arcagi(config)
    
    # Map global puzzle identifiers
    num_identifiers = config.puzzle_identifiers_start  # 0 is blank, start at 1
    identifier_map = {}
    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1
    print (f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Save
    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        # Translational augmentations
        enable_translational_augment = split_name == "train"

        # Statistics
        total_examples = 0
        total_puzzles = 0
        total_groups = 0
        
        for subset_name, subset in split.items(): # "all" is the only subset
            # --- MODIFICATION ---
            # Construct subset
            results = {k: [] for k in ["inputs", "labels", "intermediate_steps", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            # --- END MODIFICATION ---
            
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)
            
            example_id = 0
            puzzle_id = 0
            
            for group in subset:
                for puzzle in group:
                    # Push puzzle
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    for _idx_ex, ex_data in enumerate(puzzle.examples):
                        # --- MODIFICATION ---
                        inp, out, steps = np_grid_to_seq_translational_augment(
                            ex_data["input"], 
                            ex_data["output"], 
                            ex_data["steps"], 
                            do_translation=enable_translational_augment and _idx_ex != no_aug_id
                        )
                        # --- END MODIFICATION ---
                            
                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        results["intermediate_steps"].append(steps) # Save the stacked steps
                        example_id += 1
                        
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(identifier_map[puzzle.id])
                    
                    puzzle_id += 1
                    total_puzzles += 1
                    
                # Push group
                results["group_indices"].append(puzzle_id)
                total_groups += 1
            
            for k, v in results.items():
                if k in {"inputs", "labels", "intermediate_steps"}: # Stack steps too
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)
                
                np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), v)
        
        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=ARCMaxGridSize * ARCMaxGridSize,
            vocab_size=10 + 2,  # PAD + EOS + "0" ... "9"
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            total_puzzles=total_puzzles,
            sets=list(split.keys())
        )

        # Save metadata as JSON.
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            
    # Save IDs mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)
    
    # Save Test Puzzles
    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
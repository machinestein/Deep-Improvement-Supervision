from typing import List, Tuple, Dict
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
    
    # puzzle_identifiers_start is now used again
    puzzle_identifiers_start: int = 1 # start > 1 to handle multiple datasets
    
    # Number of (2 train, 1 test) combinations to sample per puzzle.
    num_combinations_per_puzzle: int = 1
    
ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5

PuzzleIdSeparator = "|||"

# Define sequence lengths
GRID_SEQ_LEN = ARCMaxGridSize * ARCMaxGridSize
    

@dataclass
class ARCPuzzle:
    id: str
    train_examples: List[Tuple[np.ndarray, np.ndarray]]
    test_examples: List[Tuple[np.ndarray, np.ndarray]]

    
def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)

    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def grid_to_seq(grid: np.ndarray):
    """
    Converts a single grid to a padded, flattened sequence with EOS markers.
    No translational augmentation.
    """
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # No random translation
    pad_r = pad_c = 0

    # Pad grid
    nrow, ncol = grid.shape
    grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)), constant_values=0)

    # Add <eos>
    eos_row, eos_col = pad_r + nrow, pad_c + ncol
    if eos_row < ARCMaxGridSize:
        grid[eos_row, pad_c:eos_col] = 1
    if eos_col < ARCMaxGridSize:
        grid[pad_r:eos_row, eos_col] = 1

    return grid.flatten()


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: ARCPuzzle):
    # Hash the puzzle for checking equivalence
    hashes = []
    for input, label in puzzle.train_examples:
        hashes.append(f"train|{grid_hash(input)}|{grid_hash(label)}")
    for input, label in puzzle.test_examples:
        hashes.append(f"test|{grid_hash(input)}|{grid_hash(label)}")
            
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


def convert_single_arc_puzzle(results: dict, name: str, puzzle: dict, aug_count: int, dest: Tuple[str, str]):
    """
    MODIFIED: This function now takes a single destination `dest` for the entire puzzle.
    It populates the .train_examples and .test_examples lists identically for ALL splits.
    """
    
    # Get train and test examples separately
    train_pairs = [(arc_grid_to_np(ex["input"]), arc_grid_to_np(ex["output"])) for ex in puzzle.get("train", [])]
    test_pairs = [(arc_grid_to_np(ex["input"]), arc_grid_to_np(ex.get("output", [[0]]))) for ex in puzzle.get("test", [])]

    dest_split, dest_set = dest
    
    # --- CORRECTED UNIFIED LOGIC ---
    # ALL puzzles get the same treatment.
    # .train_examples = ONLY train pairs (for 2-shot sampling)
    # .test_examples = ONLY test pairs (for 1-shot sampling)
    base_puzzle = ARCPuzzle(name, train_examples=train_pairs, test_examples=test_pairs)
    # --- END CORRECTED LOGIC ---

    group = [base_puzzle] # This group contains just the original puzzle
    
    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(base_puzzle)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, _map_grid = aug(name)

            # Augment based on the base puzzle's lists
            aug_train = [(_map_grid(inp), _map_grid(label)) for (inp, label) in base_puzzle.train_examples]
            aug_test = [(_map_grid(inp), _map_grid(label)) for (inp, label) in base_puzzle.test_examples]
            augmented_puzzle = ARCPuzzle(aug_name, aug_train, aug_test)

            # Check duplicate
            h = puzzle_hash(augmented_puzzle)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented_puzzle)
                
            if len(group) >= aug_count + 1:
                break
            
        if len(group) < aug_count + 1:
            print (f"[Puzzle {name}] augmentation not full, only {len(group) - 1}")

    # Append the whole group (original + augmentations) to the destination
    results.setdefault(dest_split, {})
    results[dest_split].setdefault(dest_set, [])
    results[dest_split][dest_set].append(group)


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
        # Load all puzzles in this subset
        with open(f"{config.input_file_prefix}_{subset_name}_challenges.json", "r") as f:
            puzzles = json.load(f)

        sols_filename = f"{config.input_file_prefix}_{subset_name}_solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)
                
                for puzzle_id in puzzles.keys():
                    if puzzle_id in sols:
                        for idx, sol_grid in enumerate(sols[puzzle_id]):
                            if idx < len(puzzles[puzzle_id]["test"]):
                                puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            # Fill with dummy
            print (f"{subset_name} solutions not found, filling with dummy")

            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)
        
        # Assign by fraction
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            
            # Determine the single, final destination for this entire puzzle
            puzzle_destination = None
            for f, dest in test_examples_map.get(subset_name, test_examples_map["_default"]):
                if fraction < f:
                    puzzle_destination = dest
                    break
                    
            assert puzzle_destination is not None
            
            if puzzle_destination[0] == "test":
                test_puzzles[name] = puzzle
                
            # Call convert_single_arc_puzzle with the single destination
            convert_single_arc_puzzle(results, name, puzzle, config.num_aug, puzzle_destination)
            
            total_puzzles += 1

    print (f"Total puzzles: {total_puzzles}")
    return results, test_puzzles


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)
    
    # Read dataset
    data, test_puzzles = load_puzzles_arcagi(config)
    
    # --- CONSTANTS ---
    MAX_TRAIN_EXAMPLES = 2
    MAX_TEST_EXAMPLES = 1
    MAX_EXAMPLES = MAX_TRAIN_EXAMPLES + MAX_TEST_EXAMPLES # Total 3
    # --- END CONSTANTS ---

    NEW_SEQ_LEN = GRID_SEQ_LEN * MAX_EXAMPLES # 900 * 3 = 2700
    PAD_ID = 0 # Use 0 for pad, which is also ignore_label_id
    BLANK_ID = 0
    
    # Create reusable padding sequences
    pad_input_seq = np.full(GRID_SEQ_LEN, PAD_ID, dtype=np.uint8)
    pad_label_seq = np.full(GRID_SEQ_LEN, PAD_ID, dtype=np.uint8) # Use PAD_ID as ignore_label_id
    
    # Map global puzzle identifiers
    num_identifiers = config.puzzle_identifiers_start  # 0 is blank
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
        
        # Statistics
        total_examples = 0 # Total combinations
        total_puzzles = 0  # Total ARCPuzzle objects
        total_groups = 0
        
        for subset_name, subset in split.items(): # "all" is the only subset
            # Construct subset
            results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)
            
            example_id = 0 # This now tracks total combinations
            puzzle_id = 0  # This now tracks total ARCPuzzles
            
            for group in subset:
                for puzzle in group: # e.g., "puzzle_X_aug_5"
                    
                    train_ex = puzzle.train_examples
                    test_ex = puzzle.test_examples
                    num_train = len(train_ex)
                    num_test = len(test_ex)

                    # --- This sampling logic now works correctly for all splits ---
                    for _ in range(config.num_combinations_per_puzzle):
                        all_inputs_seqs = []
                        all_labels_seqs = []

                        # --- Select exactly 2 train examples ---
                        selected_train_ex = []
                        if num_train == 0:
                            selected_train_ex.append(None) # Pad
                            selected_train_ex.append(None) # Pad
                        elif num_train == 1:
                            selected_train_ex.append(train_ex[0])
                            selected_train_ex.append(None) # Pad
                        elif num_train == 2:
                            selected_train_ex.extend(train_ex) # Perfect match
                        else: # num_train > 2: Sample
                            # *** THIS IS THE SECOND CHANGE ***
                            # Changed replace=True to replace=False to avoid sampling the same example twice
                            indices = np.random.choice(num_train, MAX_TRAIN_EXAMPLES, replace=False)
                            selected_train_ex.append(train_ex[indices[0]])
                            selected_train_ex.append(train_ex[indices[1]])
                        
                        # --- Select exactly 1 test example ---
                        selected_test_ex = []
                        if num_test == 0:
                            selected_test_ex.append(None) # Pad
                        elif num_test == 1:
                            selected_test_ex.append(test_ex[0]) # Perfect match
                        else: # num_test > 1: Sample
                            index = np.random.choice(num_test, MAX_TEST_EXAMPLES, replace=True)[0]
                            selected_test_ex.append(test_ex[index])

                        # --- Combine, convert, and pad ---
                        combined_examples = selected_train_ex + selected_test_ex
                        
                        for ex in combined_examples:
                            if ex is None: # This is a padding signal
                                all_inputs_seqs.append(pad_input_seq)
                                all_labels_seqs.append(pad_label_seq)
                            else:
                                inp, out = ex
                                all_inputs_seqs.append(grid_to_seq(inp))
                                all_labels_seqs.append(grid_to_seq(out))
                        
                        # --- Concatenate all sequences for this combination ---
                        final_input_seq = np.concatenate(all_inputs_seqs)
                        final_label_seq = (
                            np.concatenate(all_labels_seqs)
                            if all_labels_seqs
                            else np.array([], dtype=np.uint8)
                        )
                        
                        results["inputs"].append(final_input_seq)
                        results["labels"].append(final_label_seq)
                        
                        example_id += 1
                        total_examples += 1
                    
                    # --- Update puzzle-level indices (after all combinations) ---
                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(identifier_map[puzzle.id])
                    
                    puzzle_id += 1
                    total_puzzles += 1
                    
                # --- Update group-level indices (after all puzzles in group) ---
                results["group_indices"].append(puzzle_id)
                total_groups += 1
            
            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)
                
                np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), v)
        
        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=NEW_SEQ_LEN, # 2700
            vocab_size=10 + 2,  # PAD + EOS + "0" ... "9"
            pad_id=PAD_ID,
            ignore_label_id=PAD_ID, # Crucial: pad_id is also ignore_label_id
            blank_identifier_id=BLANK_ID,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles if total_puzzles > 0 else 0, # e.g. 10.0
            total_puzzles=total_puzzles,
            sets=list(split.keys())
        )

        # Save metadata as JSON.
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            
    # Save the full ID mapping
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
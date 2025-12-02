import os
import json
import numpy as np

# --- Configuration ---
# Point these to the *root* of your dataset folders
TRAIN_DATA_PATH = "data/arc_with_steps_new"
TEST_DATA_PATH = "data/arc1concept-aug-1000"
# ---------------------

def check_identifiers(data_path, split="train"):
    """Loads and analyzes puzzle identifiers for a given dataset path and split."""
    print(f"\n--- Checking Dataset: {data_path} (split: {split}) ---")
    
    # Paths
    #ids_npy_path = os.path.join(data_path, split, "puzzle_identifiers.npy")
    #meta_json_path = os.path.join(data_path, split, "dataset.json")

    ids_npy_path = os.path.join(data_path, split, "puzzle_identifiers.npy")
    meta_json_path = os.path.join(data_path, split, "dataset.json")
    
    
    if not os.path.exists(ids_npy_path) or not os.path.exists(meta_json_path):
        print(f"Error: Could not find required files in {os.path.join(data_path, split)}")
        return None, 0

    # Load metadata
    with open(meta_json_path, 'r') as f:
        metadata = json.load(f)
    
    num_from_meta = metadata.get("num_puzzle_identifiers")
    print(f"Metadata reports {num_from_meta} puzzle identifiers.")
    
    # Load identifiers
    try:
        identifiers = np.load(ids_npy_path)
    except Exception as e:
        print(f"Error loading {ids_npy_path}: {e}")
        return None, num_from_meta

    if identifiers.size == 0:
        print("Identifier file is empty.")
        return set(), num_from_meta

    unique_ids = set(np.unique(identifiers))
    min_id = np.min(identifiers)
    max_id = np.max(identifiers)
    
    print(f"Found {len(unique_ids)} unique IDs in .npy file.")
    print(f"ID Range: [{min_id}, {max_id}]")
    
    return unique_ids, num_from_meta

if __name__ == "__main__":
    print("Running Puzzle Identifier Check...")
    
    train_ids, train_meta_count = check_identifiers(TRAIN_DATA_PATH, "train")
    test_ids, test_meta_count = check_identifiers(TEST_DATA_PATH, "test")
    
    if train_ids is not None and test_ids is not None:
        print("\n--- Summary ---")
        
        # Check for overlap
        overlap = train_ids.intersection(test_ids)
        if not overlap:
            print("✅ SUCCESS: No identifier overlap found between train and test sets.")
        else:
            print(f"🚨 WARNING: Found {len(overlap)} overlapping identifiers!")
            print(f"   Example overlap: {list(overlap)[:5]}")

        # Check ranges
        train_min = min(train_ids) if train_ids else 0
        train_max = max(train_ids) if train_ids else 0
        test_min = min(test_ids) if test_ids else 0
        
        print(f"\nTrain ID Range: [{train_min}, {train_max}] (Count: {train_meta_count})")
        print(f"Test ID Range:  [{min(test_ids) if test_ids else 0}, {max(test_ids) if test_ids else 0}] (Count: {test_meta_count})")
        
        # This check will confirm the root cause
        if test_min >= train_meta_count:
            print(f"\nDIAGNOSIS: Test IDs start at {test_min}, which is >= the train count of {train_meta_count}.")
            print("This confirms the 'max_identifier_id' was clamping all your test data.")
        else:
             print(f"\nDIAGNOSIS: Test IDs start at {test_min}, which is *less than* the train count {train_meta_count}.")
             print("This might indicate a different problem if the fix doesn't work,")
             print("but the overlap/clamping is still the most likely issue.")
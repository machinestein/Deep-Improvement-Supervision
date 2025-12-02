#!/usr/bin/env python3
"""
Test Script for N-Queens Dataset Builder

This script tests the N-Queens dataset builder and visualizes some examples.
Run this to verify that the dataset generation works correctly.

Usage:
    python test_nqueens_dataset.py
"""

import numpy as np
import os
import json
import sys
import subprocess

def visualize_board(board, title="Board"):
    """
    Visualize an N-Queens board
    
    Args:
        board: NxN numpy array (0=empty, 1=queen after decoding)
        title: Title to display
    """
    N = len(board)
    print(f"\n{title} ({N}x{N}):")
    print("  " + " ".join([str(i) for i in range(N)]))
    for i, row in enumerate(board):
        row_str = " ".join(['Q' if cell == 1 else '.' for cell in row])
        print(f"{i} {row_str}")


def check_solution_validity(board):
    """
    Check if a board represents a valid N-Queens solution
    
    Args:
        board: NxN numpy array (0=empty, 1=queen)
    
    Returns:
        (is_valid, errors) tuple
    """
    N = len(board)
    errors = []
    
    # Count total queens
    num_queens = np.sum(board)
    if num_queens != N:
        errors.append(f"Expected {N} queens, found {num_queens}")
    
    # Check rows (each row should have exactly 1 queen)
    for i in range(N):
        row_sum = np.sum(board[i, :])
        if row_sum != 1:
            errors.append(f"Row {i} has {row_sum} queens (expected 1)")
    
    # Check columns (each column should have exactly 1 queen)
    for j in range(N):
        col_sum = np.sum(board[:, j])
        if col_sum != 1:
            errors.append(f"Column {j} has {col_sum} queens (expected 1)")
    
    # Check diagonals (each diagonal should have at most 1 queen)
    # Main diagonals (top-left to bottom-right)
    for d in range(-(N-1), N):
        diag_sum = np.trace(board, offset=d)
        if diag_sum > 1:
            errors.append(f"Main diagonal {d} has {diag_sum} queens (max 1)")
    
    # Anti-diagonals (top-right to bottom-left)
    flipped = np.fliplr(board)
    for d in range(-(N-1), N):
        diag_sum = np.trace(flipped, offset=d)
        if diag_sum > 1:
            errors.append(f"Anti-diagonal {d} has {diag_sum} queens (max 1)")
    
    return len(errors) == 0, errors


def test_dataset_generation():
    """
    Test the N-Queens dataset generation
    """
    print("=" * 80)
    print("N-Queens Dataset Builder Test")
    print("=" * 80)
    
    # Test configuration
    test_output_dir = "data/nqueens-test"
    
    # Create output directory if it doesn't exist
    os.makedirs(test_output_dir, exist_ok=True)
    
    print("\nStep 1: Generating small test dataset...")
    print("-" * 80)
    
    # Generate a small dataset for testing
    cmd = [
        "python3", "dataset/build_nqueens_dataset.py",
        "--board-size", "8",
        "--num-train", "10", 
        "--num-test", "5",
        "--num-aug", "2",
        "--output-dir", test_output_dir,
        "--seed", "42"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: Dataset generation failed!")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"\n✗ ERROR: python3 not found. Please make sure Python 3 is installed and in your PATH.")
        return False
    
    print("\n✓ Dataset generation completed")
    
    # Load and verify the dataset
    print("\nStep 2: Loading and verifying dataset...")
    print("-" * 80)
    
    train_dir = os.path.join(test_output_dir, "train")
    test_dir = os.path.join(test_output_dir, "test")
    
    # Check directories exist
    if not os.path.exists(train_dir):
        print(f"\n✗ ERROR: Train directory not found: {train_dir}")
        return False
    
    if not os.path.exists(test_dir):
        print(f"\n✗ ERROR: Test directory not found: {test_dir}")
        return False
    
    print(f"✓ Train directory: {train_dir}")
    print(f"✓ Test directory: {test_dir}")
    
    # Load metadata
    metadata_path = os.path.join(train_dir, "dataset.json")
    if not os.path.exists(metadata_path):
        print(f"\n✗ ERROR: Metadata file not found: {metadata_path}")
        return False
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"\n✗ ERROR: Failed to load metadata: {e}")
        return False
    
    print(f"\nMetadata:")
    print(f"  - Sequence length: {metadata.get('seq_len', 'N/A')}")
    print(f"  - Vocabulary size: {metadata.get('vocab_size', 'N/A')}")
    print(f"  - Total groups: {metadata.get('total_groups', 'N/A')}")
    print(f"  - Total puzzles: {metadata.get('total_puzzles', 'N/A')}")
    
    # Load data
    inputs_path = os.path.join(train_dir, "all__inputs.npy")
    labels_path = os.path.join(train_dir, "all__labels.npy")
    
    if not os.path.exists(inputs_path):
        print(f"\n✗ ERROR: Inputs file not found: {inputs_path}")
        return False
    if not os.path.exists(labels_path):
        print(f"\n✗ ERROR: Labels file not found: {labels_path}")
        return False
    
    try:
        inputs = np.load(inputs_path)
        labels = np.load(labels_path)
    except Exception as e:
        print(f"\n✗ ERROR: Failed to load data files: {e}")
        return False
    
    print(f"\nData shapes:")
    print(f"  - Inputs: {inputs.shape}")
    print(f"  - Labels: {labels.shape}")
    
    # Verify vocabulary
    unique_input_values = np.unique(inputs)
    unique_label_values = np.unique(labels)
    
    print(f"\nVocabulary check:")
    print(f"  - Input values: {unique_input_values}")
    print(f"  - Label values: {unique_label_values}")
    print(f"  - Expected: [0 (PAD), 1 (empty), 2 (queen)]")
    
    # Visualize some examples
    print("\nStep 3: Visualizing examples...")
    print("-" * 80)
    
    seq_len = metadata.get('seq_len', 64)  # Default to 64 for 8x8 board
    N = int(np.sqrt(seq_len))
    
    for i in range(min(3, len(inputs))):
        print(f"\nExample {i+1}:")
        
        # Decode: subtract 1 to get back to 0=empty, 1=queen
        input_board = (inputs[i] - 1).reshape(N, N)
        label_board = (labels[i] - 1).reshape(N, N)
        
        visualize_board(input_board, f"Input (Example {i+1})")
        visualize_board(label_board, f"Label (Example {i+1})")
        
        # Verify solution
        is_valid, errors = check_solution_validity(label_board)
        
        if is_valid:
            print(f"  ✓ Solution is VALID")
        else:
            print(f"  ✗ Solution is INVALID:")
            for error in errors:
                print(f"    - {error}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    # Check all solutions
    all_valid = True
    for i in range(len(labels)):
        label_board = (labels[i] - 1).reshape(N, N)
        is_valid, _ = check_solution_validity(label_board)
        if not is_valid:
            all_valid = False
            print(f"✗ Solution {i} is invalid")
    
    if all_valid:
        print(f"✓ All {len(labels)} solutions are valid!")
    else:
        print(f"✗ Some solutions are invalid")
    
    print(f"✓ Dataset generation test completed")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_dataset_generation()
    sys.exit(0 if success else 1)
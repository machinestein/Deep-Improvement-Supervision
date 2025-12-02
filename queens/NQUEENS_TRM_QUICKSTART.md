# N-Queens Training with Tiny Recursive Models (TRM) - Quick Start Guide

This guide provides a comprehensive walkthrough for training the Tiny Recursive Model on the N-Queens problem.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Explanation](#detailed-explanation)
5. [Customization](#customization)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### What is TRM?

The **Tiny Recursive Model (TRM)** is a recursive reasoning approach that achieves impressive results on hard reasoning tasks using a small neural network (only 7M parameters). Instead of relying on massive language models, TRM uses recursive reasoning to iteratively improve its predictions.

### What is N-Queens?

The **N-Queens problem** is a classic constraint satisfaction puzzle: place N chess queens on an N×N chessboard such that no two queens threaten each other (no two queens share the same row, column, or diagonal).

### Why TRM for N-Queens?

N-Queens is an excellent test case for recursive reasoning because:
- It requires constraint satisfaction (checking multiple conditions)
- Solutions can be iteratively refined
- It benefits from multiple reasoning steps
- It's a well-defined problem with clear success criteria

---

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **CUDA**: 12.6.0 or compatible (for GPU training)
- **GPU**: Recommended (training on CPU is very slow)
  - Single GPU: Works fine (12-24 hours for 8×8 board)
  - Multi-GPU: Faster training (use `torchrun`)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk Space**: 5GB for code, datasets, and checkpoints

### Software Dependencies

All dependencies are listed in `requirements.txt` and will be installed automatically.

Key packages:
- PyTorch (deep learning framework)
- Hydra (configuration management)
- Weights & Biases (optional, for experiment tracking)
- AdamATan2 (custom optimizer)

---

## Quick Start

### Option 1: Run the Complete Pipeline (Recommended)

```bash
# Navigate to the repository
cd /home/ubuntu/TinyRecursiveModels

# Run the complete pipeline script
bash /home/ubuntu/run_nqueens_trm.sh
```

This single command will:
1. Set up the environment
2. Generate the N-Queens dataset
3. Train the TRM model
4. Save checkpoints and logs

### Option 2: Step-by-Step Execution

If you prefer to run each step manually:

#### Step 1: Setup Environment

```bash
cd /home/ubuntu/TinyRecursiveModels

# Install dependencies
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2

# Optional: Setup Weights & Biases
wandb login YOUR-API-KEY
# Or disable it:
export WANDB_MODE=disabled
```

#### Step 2: Generate Dataset

```bash
# Generate 8×8 N-Queens dataset
python dataset/build_nqueens_dataset.py \
  --board-size 8 \
  --num-train 800 \
  --num-test 200 \
  --num-aug 8 \
  --output-dir data/nqueens-8x8-1k-aug-8 \
  --seed 42
```

**What this does:**
- Generates 800 training puzzles and 200 test puzzles
- Each puzzle is augmented 8 times (rotations + reflections)
- Total training examples: 800 × 9 = 7,200
- Total test examples: 200 × 1 = 200

#### Step 3: Train the Model

```bash
# Single GPU training
python pretrain.py \
  arch=trm \
  data_paths="[data/nqueens-8x8-1k-aug-8]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  +run_name=nqueens_trm_8x8 \
  ema=True
```

**For multi-GPU training (4 GPUs):**

```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/nqueens-8x8-1k-aug-8]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  +run_name=nqueens_trm_8x8 \
  ema=True
```

---

## Detailed Explanation

### Dataset Format

The N-Queens dataset follows TRM's expected format:

**Vocabulary:**
- `0`: PAD token (unused in N-Queens)
- `1`: Empty cell
- `2`: Queen

**Input:** Partial or empty N-Queens board (flattened to 1D sequence)
- For 8×8 board: 64-element sequence
- Example: `[1, 1, 2, 1, 1, 1, 1, 1, ...]` means a queen at position 2

**Label:** Complete valid solution
- Same format as input but with all N queens correctly placed

**Augmentation:**
- Each puzzle is augmented using 8 dihedral transformations:
  - Identity (no change)
  - 90° rotation
  - 180° rotation
  - 270° rotation
  - Horizontal flip
  - Vertical flip
  - Transpose (main diagonal reflection)
  - Anti-diagonal reflection

### TRM Architecture

TRM uses a recursive reasoning process:

```
For K improvement steps:
    1. Update latent state z (L_cycles times)
       z = f(x, y, z)  where x=input, y=current answer
    
    2. Update answer y based on latent z
       y = g(y, z)
    
    3. Repeat
```

**Key Hyperparameters:**

- **L_layers** (2): Number of transformer/MLP layers
  - More layers = more capacity but slower training
  
- **H_cycles** (3): Number of high-level answer improvement iterations
  - More cycles = more reasoning steps
  - Think of this as "how many times to revise the answer"
  
- **L_cycles** (6): Number of low-level latent updates per H-cycle
  - More cycles = deeper reasoning per iteration
  - Think of this as "how much to think before revising"

### Training Process

**Epochs:** The model trains for 50,000 epochs
- One epoch = one pass through the entire dataset
- With 7,200 training examples and batch size ~100, this is ~3.6M gradient updates

**Learning Rate:** 1e-4 with cosine annealing
- Starts at 1e-4
- Gradually decreases to near zero
- Helps the model converge smoothly

**Weight Decay:** 1.0 (strong L2 regularization)
- Prevents overfitting
- Keeps model weights small
- Important for small datasets

**EMA (Exponential Moving Average):** Enabled
- Maintains a running average of model weights
- Provides more stable predictions
- Often improves final performance

### Evaluation

The model is evaluated every 5,000 epochs on the test set.

**Metrics to watch:**
- **train_loss**: Should decrease steadily
- **test_loss**: Should decrease but may plateau
- **test_accuracy**: Percentage of correctly solved puzzles
  - 100% = perfect solver
  - 80%+ = very good
  - 50%+ = learning something useful

---

## Customization

### Different Board Sizes

**4×4 (Easy):**
```bash
python dataset/build_nqueens_dataset.py \
  --board-size 4 \
  --num-train 200 \
  --num-test 50 \
  --output-dir data/nqueens-4x4
```

**10×10 (Hard):**
```bash
python dataset/build_nqueens_dataset.py \
  --board-size 10 \
  --num-train 2000 \
  --num-test 500 \
  --output-dir data/nqueens-10x10
```

**12×12 (Very Hard):**
```bash
python dataset/build_nqueens_dataset.py \
  --board-size 12 \
  --num-train 5000 \
  --num-test 1000 \
  --output-dir data/nqueens-12x12
```

### Difficulty Levels

Control how many queens are given in the input:

**Empty board (hardest):**
```bash
python dataset/build_nqueens_dataset.py \
  --board-size 8 \
  --num-given-queens 0
```

**Half-filled (medium):**
```bash
python dataset/build_nqueens_dataset.py \
  --board-size 8 \
  --num-given-queens 4
```

**Almost complete (easiest):**
```bash
python dataset/build_nqueens_dataset.py \
  --board-size 8 \
  --num-given-queens 7
```

### Model Capacity

**Smaller model (faster, less accurate):**
```bash
python pretrain.py \
  arch=trm \
  arch.L_layers=1 \
  arch.H_cycles=2 \
  arch.L_cycles=4 \
  ...
```

**Larger model (slower, more accurate):**
```bash
python pretrain.py \
  arch=trm \
  arch.L_layers=3 \
  arch.H_cycles=5 \
  arch.L_cycles=8 \
  ...
```

### Training Duration

**Quick test (1 hour):**
```bash
python pretrain.py \
  epochs=5000 \
  eval_interval=1000 \
  ...
```

**Long training (better results):**
```bash
python pretrain.py \
  epochs=100000 \
  eval_interval=10000 \
  ...
```

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size: Add `global_batch_size=64` to training command
- Use smaller model: Reduce `L_layers`, `H_cycles`, or `L_cycles`
- Use gradient accumulation: Add `gradient_accumulation_steps=2`

**2. Training is very slow**

**Solutions:**
- Ensure you're using GPU: Check `nvidia-smi`
- Use multi-GPU training with `torchrun`
- Reduce dataset size for testing
- Disable W&B: `export WANDB_MODE=disabled`

**3. Model not learning (loss not decreasing)**

**Solutions:**
- Check dataset was generated correctly: `ls data/nqueens-*/train/`
- Increase learning rate: Try `lr=5e-4`
- Reduce weight decay: Try `weight_decay=0.1`
- Increase model capacity: More layers or cycles

**4. Import errors**

```
ModuleNotFoundError: No module named 'adam_atan2'
```

**Solutions:**
- Install missing package: `pip install --no-cache-dir --no-build-isolation adam-atan2`
- Check Python version: `python3 --version` (should be 3.10+)
- Reinstall requirements: `pip install -r requirements.txt`

**5. Dataset generation fails**

```
Error in solve_nqueens_backtrack
```

**Solutions:**
- Check board size is valid: 4, 6, 8, 10, 12, etc.
- Ensure enough disk space: `df -h`
- Try smaller dataset first: `--num-train 100`

---

## File Structure

After running the pipeline, you'll have:

```
TinyRecursiveModels/
├── dataset/
│   ├── build_nqueens_dataset.py    # N-Queens dataset builder (NEW)
│   ├── build_sudoku_dataset.py     # Sudoku dataset builder
│   └── common.py                    # Shared utilities
├── data/
│   └── nqueens-8x8-1k-aug-8/       # Generated N-Queens dataset
│       ├── train/
│       │   ├── dataset.json         # Metadata
│       │   ├── all__inputs.npy      # Training inputs
│       │   ├── all__labels.npy      # Training labels
│       │   └── ...
│       └── test/
│           └── ...
├── outputs/
│   └── nqueens_trm_8x8/            # Training outputs
│       ├── checkpoints/             # Model checkpoints
│       └── logs/                    # Training logs
├── pretrain.py                      # Main training script
└── ...
```

---

## Performance Expectations

### 8×8 N-Queens

**With recommended settings:**
- Training time: 12-24 hours (single GPU)
- Expected accuracy: 70-90% on test set
- Model size: ~7M parameters

### 10×10 N-Queens

**With increased capacity:**
- Training time: 24-48 hours
- Expected accuracy: 50-70% on test set
- Requires more data and deeper model

### 12×12 N-Queens

**Very challenging:**
- Training time: 48+ hours
- Expected accuracy: 30-50% on test set
- Requires significant compute and data

---

## Advanced Topics

### Using Different Architectures

TRM supports multiple architectures:

**Hierarchical Recursive Model (HRM):**
```bash
python pretrain.py arch=hrm ...
```

**Transformer Baseline:**
```bash
python pretrain.py arch=transformers_baseline ...
```

### Custom Loss Functions

Modify `models/losses.py` to implement custom loss functions for N-Queens.

### Visualization

To visualize N-Queens solutions, you can write a simple script:

```python
import numpy as np

# Load a solution
labels = np.load('data/nqueens-8x8-1k-aug-8/test/all__labels.npy')
solution = labels[0].reshape(8, 8)

# Visualize (2 = queen, 1 = empty)
for row in solution:
    print(' '.join(['Q' if cell == 2 else '.' for cell in row]))
```

---

## References

**TRM Paper:**
- Title: "Less is More: Recursive Reasoning with Tiny Networks"
- Authors: Alexia Jolicoeur-Martineau
- arXiv: https://arxiv.org/abs/2510.04871

**TRM Repository:**
- https://github.com/SamsungSAILMontreal/TinyRecursiveModels

**N-Queens Problem:**
- Wikipedia: https://en.wikipedia.org/wiki/Eight_queens_puzzle

---

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review the TRM paper for architecture details
3. Check the TRM repository issues page
4. Verify your dataset was generated correctly

---

## License

This code builds upon the TinyRecursiveModels repository, which is based on the Hierarchical Reasoning Model. Please refer to the original repositories for license information.

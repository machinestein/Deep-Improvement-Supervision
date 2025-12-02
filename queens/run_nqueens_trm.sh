#!/bin/bash

################################################################################
# COMPREHENSIVE STEP-BY-STEP GUIDE: Training TRM on N-Queens Task
################################################################################
#
# This script provides a complete walkthrough for training the Tiny Recursive
# Model (TRM) on the N-Queens problem. Each step is thoroughly documented with
# explanations of what's happening and why.
#
# Author: Generated for N-Queens TRM Training
# Date: 2025
#
################################################################################

set -e  # Exit on any error

################################################################################
# STEP 0: OVERVIEW
################################################################################
#
# The Tiny Recursive Model (TRM) is a recursive reasoning approach that uses
# a small neural network to iteratively improve predictions. We'll adapt it
# to solve N-Queens puzzles.
#
# The N-Queens problem: Place N chess queens on an NxN board such that no two
# queens threaten each other (same row, column, or diagonal).
#
# Training Pipeline:
#   1. Environment Setup
#   2. Dataset Generation
#   3. Model Training
#   4. Evaluation
#
################################################################################

echo "================================================================================"
echo "  TRM N-Queens Training Pipeline"
echo "================================================================================"
echo ""

################################################################################
# STEP 1: ENVIRONMENT SETUP
################################################################################
echo "STEP 1: Setting up environment..."
echo "--------------------------------------------------------------------------------"

# 1.1: Navigate to the TinyRecursiveModels directory
# This is where all the TRM code lives
cd /home/ubuntu/TinyRecursiveModels
echo "✓ Changed directory to TinyRecursiveModels"

# 1.2: Check Python version
# TRM requires Python 3.10 or similar
echo ""
echo "Checking Python version..."
python3 --version
echo "✓ Python version check complete"

# 1.3: Install PyTorch (if not already installed)
# TRM uses PyTorch as its deep learning framework
# NOTE: Adjust the CUDA version based on your system
# For CPU-only: pip install torch torchvision torchaudio
echo ""
echo "Installing PyTorch..."
echo "NOTE: Skipping PyTorch installation - assuming it's already installed"
echo "      If not installed, run:"
echo "      pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126"

# 1.4: Install required dependencies
# These are the packages needed by TRM
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# 1.5: Install additional optimizer
# TRM uses a custom optimizer called AdamATan2
echo ""
echo "Installing AdamATan2 optimizer..."
pip install --no-cache-dir --no-build-isolation adam-atan2 --quiet
echo "✓ AdamATan2 optimizer installed"

# 1.6: Optional - Setup Weights & Biases for experiment tracking
# W&B is used to log training metrics and visualize results
echo ""
echo "Weights & Biases setup (optional):"
echo "  To enable experiment tracking, run: wandb login YOUR-API-KEY"
echo "  To disable W&B, set: export WANDB_MODE=disabled"
export WANDB_MODE=disabled  # Disable W&B for this demo
echo "✓ W&B disabled for this run"

echo ""
echo "✓ STEP 1 COMPLETE: Environment setup finished"
echo ""

################################################################################
# STEP 2: DATASET GENERATION
################################################################################
echo "STEP 2: Generating N-Queens dataset..."
echo "--------------------------------------------------------------------------------"

# 2.1: Understanding the dataset format
# TRM expects datasets in a specific format:
#   - inputs: Partial or empty N-Queens boards (what the model sees)
#   - labels: Complete valid solutions (what the model should predict)
#   - Flattened to 1D sequences (NxN board -> N² sequence)
#   - Vocabulary: 0 (PAD), 1 (empty cell), 2 (queen)

# 2.2: Dataset configuration parameters
BOARD_SIZE=8           # Size of the chessboard (8x8 for classic 8-Queens)
NUM_TRAIN=800          # Number of training puzzles to generate
NUM_TEST=200           # Number of test puzzles to generate
NUM_AUG=8              # Augmentation factor (8 = all rotations + reflections)
OUTPUT_DIR="data/nqueens-8x8-1k-aug-8"  # Where to save the dataset

echo ""
echo "Dataset Configuration:"
echo "  - Board size: ${BOARD_SIZE}x${BOARD_SIZE}"
echo "  - Training puzzles: ${NUM_TRAIN}"
echo "  - Test puzzles: ${NUM_TEST}"
echo "  - Augmentation: ${NUM_AUG} (dihedral symmetries)"
echo "  - Output directory: ${OUTPUT_DIR}"
echo ""

# 2.3: Generate the dataset
# This runs our custom N-Queens dataset builder
echo "Generating dataset (this may take a few minutes)..."
python dataset/build_nqueens_dataset.py \
  --board-size 8 \
  --num-train 800 \
  --num-test 200 \
  --num-aug 8 \
  --output-dir "data/nqueens-8x8-1k-aug-8" \
  --seed 42

echo ""
echo "✓ STEP 2 COMPLETE: Dataset generated successfully"
echo ""

# 2.4: Verify dataset was created
echo "Verifying dataset files..."
if [ -d "${OUTPUT_DIR}/train" ] && [ -d "${OUTPUT_DIR}/test" ]; then
    echo "✓ Train and test directories created"
    echo "  Train files:"
    ls -lh ${OUTPUT_DIR}/train/*.npy | head -3
    echo "  Test files:"
    ls -lh ${OUTPUT_DIR}/test/*.npy | head -3
else
    echo "✗ ERROR: Dataset directories not found!"
    exit 1
fi
echo ""

################################################################################
# STEP 3: MODEL TRAINING
################################################################################
echo "STEP 3: Training TRM on N-Queens..."
echo "--------------------------------------------------------------------------------"

# 3.1: Understanding TRM architecture
# TRM has two main components:
#   - H (High-level) cycles: Number of answer improvement iterations
#   - L (Low-level) cycles: Number of latent state updates per improvement
#   - L_layers: Number of transformer/MLP layers
#
# The model recursively:
#   1. Updates its internal latent state (L_cycles times)
#   2. Improves its answer based on the latent state
#   3. Repeats this process (H_cycles times)

# 3.2: Training hyperparameters
# These are carefully chosen based on the TRM paper and Sudoku experiments
RUN_NAME="nqueens_trm_8x8"        # Name for this training run
ARCH="trm"                         # Architecture: TRM (Tiny Recursive Model)
EPOCHS=50000                       # Number of training epochs
EVAL_INTERVAL=5000                 # Evaluate every N epochs
LR=1e-4                           # Learning rate
WEIGHT_DECAY=1.0                  # L2 regularization strength
L_LAYERS=2                        # Number of layers in the network
H_CYCLES=3                        # Number of high-level reasoning cycles
L_CYCLES=6                        # Number of low-level latent updates

echo ""
echo "Training Configuration:"
echo "  - Run name: ${RUN_NAME}"
echo "  - Architecture: ${ARCH}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Learning rate: ${LR}"
echo "  - Weight decay: ${WEIGHT_DECAY}"
echo "  - L_layers: ${L_LAYERS}"
echo "  - H_cycles: ${H_CYCLES} (answer improvement iterations)"
echo "  - L_cycles: ${L_CYCLES} (latent state updates)"
echo ""

# 3.3: Explanation of key training arguments
echo "Key Training Arguments Explained:"
echo ""
echo "  arch=trm"
echo "    → Use the Tiny Recursive Model architecture"
echo ""
echo "  data_paths=\"[${OUTPUT_DIR}]\""
echo "    → Path to our N-Queens dataset"
echo ""
echo "  evaluators=\"[]\""
echo "    → No special evaluators (we'll use default metrics)"
echo ""
echo "  epochs=${EPOCHS}"
echo "    → Train for this many epochs (one epoch = one pass through dataset)"
echo ""
echo "  eval_interval=${EVAL_INTERVAL}"
echo "    → Evaluate model performance every N epochs"
echo ""
echo "  lr=${LR} puzzle_emb_lr=${LR}"
echo "    → Learning rate for model weights and puzzle embeddings"
echo ""
echo "  weight_decay=${WEIGHT_DECAY}"
echo "    → L2 regularization to prevent overfitting"
echo ""
echo "  arch.L_layers=${L_LAYERS}"
echo "    → Number of transformer/MLP layers in the network"
echo ""
echo "  arch.H_cycles=${H_CYCLES}"
echo "    → Number of high-level answer improvement iterations"
echo "    → More cycles = more reasoning steps = better solutions"
echo ""
echo "  arch.L_cycles=${L_CYCLES}"
echo "    → Number of low-level latent state updates per H-cycle"
echo "    → More cycles = deeper reasoning per iteration"
echo ""
echo "  ema=True"
echo "    → Use Exponential Moving Average for more stable training"
echo ""
echo "  +run_name=${RUN_NAME}"
echo "    → Name for this experiment (for logging and checkpoints)"
echo ""

# 3.4: Start training
echo "Starting training..."
echo "NOTE: This will take several hours depending on your hardware"
echo "      - With GPU: ~12-24 hours"
echo "      - With CPU: Much longer (not recommended)"
echo ""
echo "Training progress will be displayed below..."
echo "--------------------------------------------------------------------------------"
echo ""

# 3.5: Run the training command
# For single GPU training:
python pretrain.py \
  arch=${ARCH} \
  data_paths="[${OUTPUT_DIR}]" \
  evaluators="[]" \
  epochs=${EPOCHS} \
  eval_interval=${EVAL_INTERVAL} \
  lr=${LR} \
  puzzle_emb_lr=${LR} \
  weight_decay=${WEIGHT_DECAY} \
  puzzle_emb_weight_decay=${WEIGHT_DECAY} \
  arch.L_layers=${L_LAYERS} \
  arch.H_cycles=${H_CYCLES} \
  arch.L_cycles=${L_CYCLES} \
  +run_name=${RUN_NAME} \
  ema=True

# For multi-GPU training (e.g., 4 GPUs), use this instead:
# torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
#   arch=${ARCH} \
#   data_paths="[${OUTPUT_DIR}]" \
#   evaluators="[]" \
#   epochs=${EPOCHS} \
#   eval_interval=${EVAL_INTERVAL} \
#   lr=${LR} \
#   puzzle_emb_lr=${LR} \
#   weight_decay=${WEIGHT_DECAY} \
#   puzzle_emb_weight_decay=${WEIGHT_DECAY} \
#   arch.L_layers=${L_LAYERS} \
#   arch.H_cycles=${H_CYCLES} \
#   arch.L_cycles=${L_CYCLES} \
#   +run_name=${RUN_NAME} \
#   ema=True

echo ""
echo "✓ STEP 3 COMPLETE: Training finished"
echo ""

################################################################################
# STEP 4: POST-TRAINING
################################################################################
echo "STEP 4: Post-training information..."
echo "--------------------------------------------------------------------------------"

# 4.1: Where are the checkpoints?
echo ""
echo "Model checkpoints are saved in:"
echo "  outputs/${RUN_NAME}/"
echo ""

# 4.2: How to evaluate the model
echo "To evaluate the trained model:"
echo "  1. The model is automatically evaluated during training"
echo "  2. Check the training logs for accuracy metrics"
echo "  3. Look for 'test_accuracy' in the output"
echo ""

# 4.3: Understanding the metrics
echo "Key Metrics to Watch:"
echo "  - train_loss: How well the model fits the training data"
echo "  - test_loss: How well the model generalizes to unseen data"
echo "  - test_accuracy: Percentage of correctly solved N-Queens puzzles"
echo ""

# 4.4: Experiment tracking
echo "Experiment Tracking:"
echo "  - If W&B is enabled, view results at: https://wandb.ai/"
echo "  - Logs are also saved locally in: outputs/${RUN_NAME}/"
echo ""

################################################################################
# STEP 5: CUSTOMIZATION OPTIONS
################################################################################
echo "STEP 5: Customization options..."
echo "--------------------------------------------------------------------------------"
echo ""
echo "To experiment with different configurations:"
echo ""
echo "1. Different board sizes:"
echo "   Change BOARD_SIZE to 4, 6, 10, etc."
echo "   Larger boards are harder and may need more training"
echo ""
echo "2. More training data:"
echo "   Increase NUM_TRAIN (e.g., 2000, 5000)"
echo "   More data usually improves generalization"
echo ""
echo "3. Deeper reasoning:"
echo "   Increase H_CYCLES (e.g., 5, 7)"
echo "   More cycles allow more iterative improvements"
echo ""
echo "4. Larger model:"
echo "   Increase L_LAYERS (e.g., 3, 4)"
echo "   Bigger model can learn more complex patterns"
echo ""
echo "5. Different difficulty:"
echo "   Modify num_given_queens in dataset builder"
echo "   - 0: Empty board (hardest)"
echo "   - N-1: Almost complete (easiest)"
echo ""
echo "6. Multi-GPU training:"
echo "   Uncomment the torchrun command in STEP 3.5"
echo "   Adjust --nproc-per-node to your GPU count"
echo ""

################################################################################
# FINAL NOTES
################################################################################
echo "================================================================================"
echo "  Training Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  ✓ Environment configured"
echo "  ✓ Dataset generated (${NUM_TRAIN} train + ${NUM_TEST} test puzzles)"
echo "  ✓ Model trained for ${EPOCHS} epochs"
echo "  ✓ Checkpoints saved to outputs/${RUN_NAME}/"
echo ""
echo "Next Steps:"
echo "  1. Check training logs for final accuracy"
echo "  2. Experiment with different hyperparameters"
echo "  3. Try larger board sizes (10x10, 12x12)"
echo "  4. Analyze which puzzles the model solves vs. struggles with"
echo ""
echo "For questions or issues, refer to:"
echo "  - TRM Paper: https://arxiv.org/abs/2510.04871"
echo "  - TRM Repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels"
echo ""
echo "================================================================================"

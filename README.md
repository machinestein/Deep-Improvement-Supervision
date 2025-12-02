# Deep Improvement Supervision

Recent work has demonstrated that small, looped architectures, such as Tiny Recursive Models (TRMs), can outperform Large Language Models (LLMs) on complex reasoning tasks, including the Abstraction and Reasoning Corpus (\texttt{ARC}). In this work, we investigate a core question: how can we further improve the efficiency of these methods with minimal changes? To address this, we frame the asymmetric latent reasoning of TRMs as both an implicit policy improvement algorithm and a form of classifier-free diffusion guidance. Building on these insights, we propose a novel training scheme that provides a target for each loop during training. We demonstrate that our approach significantly enhances training efficiency. Our method reduces the total number of forward passes by 18× and eliminates halting mechanisms, while maintaining quality comparable to standard TRMs. Notably, we achieve 24$\%$ accuracy on ARC-1 with only 0.8M parameters, outperforming most LLMs.


This is based on the the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". [Paper](https://arxiv.org/abs/2510.04871). Also this code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).TRM is a recursive reasoning approach that using a tiny 7M parameters neural network.

### How TRM works
Tiny Recursion Model (TRM) recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y and latent z. For up to K improvements steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting.

### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2


## Experiments

### Assume training of compact model on ARC-AGI-1:

```bash
torchrun --nproc_per_node=NUM_GPUS train_arc_tiny_corrupt.py

```

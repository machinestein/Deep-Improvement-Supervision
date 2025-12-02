# [Deep Improvement Supervision](https://arxiv.org/pdf/2511.16886)

### Abstract
Recent work has demonstrated that small, looped architectures, such as Tiny Recursive Models (TRMs), can outperform Large Language Models (LLMs) on complex reasoning tasks, including the Abstraction and Reasoning Corpus ARC. In this work, we investigate a core question: how can we further improve the efficiency of these methods with minimal changes? To address this, we propose a novel training scheme that provides a target for each loop during training. Our method reduces the total number of forward passes by 18× and eliminates halting mechanisms, while maintaining quality comparable to standard TRMs. Notably, we achieve 24% accuracy on ARC-1 with only 0.8M parameters.


### Method

**Deep Improvement Supervision (DIS)**. While the standard Tiny Recursive Model  relies on a **black-box** recurrence where the model must implicitly discover how to improve, DIS explicitly enforces a positive Advantage Margin at every recursive step. We achieve this by constructing a sequence of intermediate targets that strictly contract toward the solution. Our results are grounded in analyses that formally frame the latent reasoning of TRMs as both an $\color{red}{\textbf{implicit policy improvement}}$  algorithm and a form of $\color{Orange}{\textbf{classifier-free diffusion guidance}}$.

$\color{Cyan}{\textbf{Discrete Diffusion Targets}}$ Instead of supervising only the final output, DIS uses a target generator—specifically a discrete diffusion-style schedule—that produces a sequence of intermediate answers. The model is trained to match the reference logits to the previous target and the conditional logits to the improved target. This forces the residual update to explicitly encode the transition toward the solution, effectively turning the reasoning process into a verifiable, step-wise policy improvement scheme. Efficiency Gains By using a fixed number of refinement steps (e.g., 6 steps) with explicit targets, we eliminate the need for the learned halting mechanism and the "no-grad" warm-up cycles required by the original TRM. 

### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Run model

```bash
## Prepare datasets

## ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

## ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2


## Experiments

## To run DIS training procedure use the following command.
## Assume training of compact model (0.8M) on ARC-AGI-1 (modify config class for different settings).

torchrun --nproc_per_node=NUM_GPUS train_arc_tiny_corrupt.py 
```
### Architecture & Base
We adapted our method and code from the official codebase for "Less is More: Recursive Reasoning with Tiny Networks"  [Paper](https://arxiv.org/abs/2510.04871). As well as from the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis). 

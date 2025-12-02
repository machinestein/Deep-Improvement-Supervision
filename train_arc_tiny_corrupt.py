from typing import Optional, Any, Sequence, List
from dataclasses import dataclass, field, asdict
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
# from adam_atan2 import AdamATan2 # This was commented in the original file

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

# --- Configuration Classes (Replaces Hydra YAML files) ---

@dataclass
class LossConfig:
    """Configuration for the loss function."""
    name: str = "losses@ACTLossHeadV4"
    loss_type: str = "stablemax_cross_entropy"

@dataclass
class ArchConfig:
    """Configuration for the model architecture."""
    name: str = "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV5"
    loss: LossConfig = field(default_factory=LossConfig)
    halt_exploration_prob: float = 0.1
    halt_max_steps: int = 6
    N_supervision: int = 6 
    H_cycles: int = 1#3  # Reduced from 3
    L_cycles: int = 2#6  # Reduced from 6
    H_layers: int = 0
    L_layers: int = 1  # Reduced from 2
    hidden_size: int = 256  # Reduced from 512
    num_heads: int = 8  # Reduced from 8
    expansion: int = 4  # Reduced from 4
    puzzle_emb_ndim: int = 256  # Reduced from 512
    pos_encodings: str = "rope"
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

@dataclass
class EvaluatorConfig:
    """Configuration for an evaluator."""
    name: str

@dataclass
class PretrainConfig:
    """Main configuration for the pretraining script."""
    # Architecture
    arch: ArchConfig = field(default_factory=ArchConfig)
    
    # Data
    data_paths: List[str] = field(default_factory=lambda: ['data/arc1concept-aug-1000'])
    data_paths_test: List[str] = field(default_factory=list)
    
    # Evaluators
    evaluators: List[EvaluatorConfig] = field(default_factory=lambda: [EvaluatorConfig(name="arc@ARC")])

    # Training Hyperparameters
    global_batch_size: int = 512#768
    epochs: int = 200000
    lr: float = 1e-4
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # Puzzle Embedding Hyperparameters
    puzzle_emb_lr: float = 1e-2
    puzzle_emb_weight_decay: float = 0.1

    # Experiment Naming & Checkpointing
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_every_eval: bool = True
    
    # Evaluation
    eval_interval: Optional[int] = 5000
    min_eval_interval: int = 0
    eval_save_outputs: List[str] = field(default_factory=list)

    # Extras
    seed: int = 0
    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset_paths = config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=dataset_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Convert arch and loss dataclasses to dicts for model instantiation
    arch_dict = asdict(config.arch)
    loss_dict = arch_dict.pop('loss')
    arch_dict.pop('name')
    loss_dict.pop('name', None) # Remove 'name' key as it's not an argument for the loss head constructor

    model_cfg = dict(
        **arch_dict,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **loss_dict)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            # AdamATan2 is not defined, assuming a placeholder or a different optimizer might be used.
            # Using AdamW as a standard replacement.
            torch.optim.AdamW(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
             # AdamATan2 is not defined, using AdamW instead.
            torch.optim.AdamW(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}.pth"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding shape. Found {puzzle_emb.shape}, Expected {expected_shape}")
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata
            )
            evaluators.append(cls)
    return evaluators


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    batch = {k: v.cuda() for k, v in batch.items()}
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    ((1 / global_batch_size) * loss).backward()

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics.get("count", 1), 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, evaluators: List[Any], rank: int, world_size: int, cpu_group: Optional[dist.ProcessGroup]):
    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        save_preds, metric_keys, metric_values = {}, [], None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)

            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=return_keys)
                inference_steps += 1
                if all_finish:
                    break
            
            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, []).append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}
        if config.checkpoint_path and len(save_preds):
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}.pth"))

        reduced_metrics = None
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            if rank == 0:
                reduced_metrics_np = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {metric_name: reduced_metrics_np[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                    for set_id, set_name in enumerate(set_ids)
                }
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
            evaluator_save_path = None
            if config.checkpoint_path:
                evaluator_save_path = os.path.join(config.checkpoint_path, f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}")
                os.makedirs(evaluator_save_path, exist_ok=True)
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics:
                if reduced_metrics is None: reduced_metrics = {}
                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
        if rank == 0: print("All evaluators completed!")
    return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)

    code_list = [get_model_source_path(config.arch.name), get_model_source_path(config.arch.loss.name)]
    for code_file in code_list:
        if code_file and os.path.exists(code_file):
            shutil.copy(code_file, os.path.join(config.checkpoint_path, os.path.basename(code_file)))

    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "wt") as f:
        yaml.dump(asdict(config), f)
    
    wandb.run.log_code(config.checkpoint_path)


def launch():
    RANK, WORLD_SIZE, CPU_PROCESS_GROUP = 0, 1, None
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    # Load and sync config across processes
    objects = [None]
    if RANK == 0:
        config = PretrainConfig()
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
        objects = [config]
    if WORLD_SIZE > 1:
        dist.broadcast_object_list(objects, src=0)
    config: PretrainConfig = objects[0]

    torch.random.manual_seed(config.seed + RANK)

    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader, eval_metadata, evaluators = None, None, []
    try:
        eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        print(f"Could not create evaluation dataloader or evaluators: {e}")

    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    progress_bar, ema_helper = None, None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=asdict(config), settings=wandb.Settings(_disable_stats=True))
        wandb.log({"num_params": sum(p.numel() for p in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}]: Epoch {_iter_id * train_epochs_per_iter}")
        train_state.model.train()
        for _, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
            if RANK == 0 and metrics:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval and eval_loader is not None:
            train_state_eval = train_state
            if config.ema:
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            
            train_state_eval.model.eval()
            metrics = evaluate(config, train_state_eval, eval_loader, eval_metadata, evaluators, rank=RANK, world_size=WORLD_SIZE, cpu_group=CPU_PROCESS_GROUP)
            if RANK == 0 and metrics:
                wandb.log(metrics, step=train_state.step)
            
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)
            if config.ema:
                del train_state_eval

    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        wandb.finish()


if __name__ == "__main__":
    # Ensure you have the required dependencies (puzzle_dataset, utils, models) in your PYTHONPATH
    # Example: export PYTHONPATH=$PYTHONPATH:/path/to/your/project
    # Then run: python pretrain.py
    # For distributed training: torchrun --nproc_per_node=NUM_GPUS train_arc_tiny_corrupt.py
    launch()


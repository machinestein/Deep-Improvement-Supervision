from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

def corrupt_linear_path(x_0: torch.Tensor, x_1: torch.Tensor, num_steps: int) -> List[torch.Tensor]:
    """Generates a sequence of corrupted grids interpolating between input (x_0) and target (x_1)."""
    if x_0.shape != x_1.shape:
        raise ValueError("Input and target tensors must have the same shape.")

    # Works with both 1D (flattened) and 2D tensors
    diff_mask = (x_0 != x_1).squeeze()
    diff_indices = torch.where(diff_mask)
    num_diff = len(diff_indices[0])

    if num_diff == 0:
        return [x_0.clone() for _ in range(num_steps)]

    frames = []
    perm = torch.randperm(num_diff)
    permuted_diff_indices = tuple(d[perm] for d in diff_indices)

    for k in range(num_steps):
        t = k / (num_steps - 1)
        num_to_replace = int(t * num_diff)
        
        current_frame = x_0.clone()
        if num_to_replace > 0:
            indices_to_replace = tuple(d[:num_to_replace] for d in permuted_diff_indices)
            # Create a view for replacement
            target_view = x_1.squeeze()
            frame_view = current_frame.squeeze()
            frame_view[indices_to_replace] = target_view[indices_to_replace]
        
        frames.append(current_frame)
    return frames, None

def batch_corrupt_linear_path(x_0_batch: torch.Tensor, x_1_batch: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Generates corruption paths for a whole batch of inputs and targets in a traceable way."""
    batch_size = x_0_batch.shape[0]
    seq_len = x_0_batch.shape[1]
    
    all_paths = torch.empty(
        (batch_size, num_steps, seq_len),
        dtype=x_0_batch.dtype,
        device=x_0_batch.device
    )
    
    # This loop is over a static range, which torch.compile can handle.
    for i in range(batch_size):
        frames, _ = corrupt_linear_path(
            x_0=x_0_batch[i],
            x_1=x_1_batch[i],
            num_steps=num_steps
        )
        if frames:
            all_paths[i] = torch.stack(frames).squeeze(1)
    return all_paths

def batch_corrupt_linear_path_v2(x0: torch.Tensor, x1: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Compile‑friendly, vectorized corruption path builder.
    x0, x1: [B, L] integer tensors (start and target)
    Returns: [B, num_steps, L] integer tensor of intermediate frames.
    """
    assert x0.shape == x1.shape and x0.ndim == 2, "x0 and x1 must be [B, L]"
    B, L = x0.shape
    device = x0.device

    if num_steps <= 1:
        return x0[:, None, :].clone()

    diff = (x0 != x1)                              # [B, L] bool
    n_diff = diff.sum(dim=1, dtype=torch.long)     # [B]

    # Randomize order across differing positions; place non-diff at the end.
    scores = torch.rand(B, L, device=device)       # float
    scores = torch.where(diff, scores, torch.full_like(scores, float('inf')))
    perm = torch.argsort(scores, dim=1)            # [B, L]
    rank = torch.argsort(perm, dim=1)              # [B, L] rank of each original pos

    s = torch.arange(num_steps, device=device, dtype=torch.long)   # [S]
    # number of replaced positions at each step for each batch element
    replaced = (s[None, :] * n_diff[:, None]) // (num_steps - 1)   # [B, S]
    mask = rank[:, None, :] < replaced[:, :, None]                 # [B, S, L] bool

    paths = torch.where(mask, x1[:, None, :], x0[:, None, :])      # [B, S, L]
    return paths


@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


@dataclass
class TinyRecursiveReasoningModel_ACTV3Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]
    intermediate_labels_path: torch.Tensor 
    
class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    # --- NEW: classifier-free guidance + guidance weight ---
    cf_uncond_prob: float = 0.10     # P(o = ∅) during TRAIN (per-sample dropout)
    cfg_guidance_w: float = 2.0  

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        #Time-added
        max_steps = getattr(self.config, "halt_max_steps", None)
        if max_steps is None:
            max_steps = getattr(self.config, "N_supervision", 16)
        
        self.timestep_embed = CastedEmbedding(
            max_steps + 2,                   # small cushion (0, 1..max, plus guard)
            self.config.hidden_size,
            #batch_size=self.config.batch_size,
            init_std=1.0 / math.sqrt(self.config.hidden_size),  # same scale as others
            cast_to=self.forward_dtype
        )
        #-----

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        #self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        #self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        h_init_tensor = trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        self.register_buffer('H_init', h_init_tensor, persistent=True)
        l_init_tensor = trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        self.register_buffer('L_init', l_init_tensor, persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    #def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
    def _input_embeddings(
        self,
        input: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
        timestep_idxs: Optional[torch.Tensor] = None):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        #Time-added
        if timestep_idxs is not None:
            # idxs: [B] -> [B, H] -> broadcast as [B, L_total, H]
            t_emb = self.timestep_embed(timestep_idxs.to(torch.int32))          # [B, H]
            t_emb = t_emb[:, None, :].expand(-1, embedding.size(1), -1)         # [B, L', H]
            embedding = embedding + t_emb
        #-----
        
        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    # def empty_carry(self, batch_size: int):
    #     return TinyRecursiveReasoningModel_ACTV1InnerCarry(
    #         z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
    #         z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
    #     )
        
    def empty_carry(self, batch_size: int):
        # Get the device from a registered buffer (like self.H_init)
        device = self.H_init.device
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        device = self.H_init.device
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    #Time-added function
    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
        timestep_idxs: Optional[torch.Tensor] = None):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
    
        # NOTE the extra argument here:
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"],
            timestep_idxs=timestep_idxs)
        
        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])



class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


@dataclass
class TinyRecursiveReasoningModel_ACTV5Carry:
    """
    Carry state for V5. Identical to V3Carry but adds
    `prev_prediction` to store the model's own output from the
    previous step, which is used as the input for the current step.
    """
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]
    intermediate_labels_path: torch.Tensor 
    prev_prediction: torch.Tensor # Holds pred_{t-1}


class TinyRecursiveReasoningModel_ACTV5(nn.Module):
    """
    ACT wrapper.
    V5 modification:
    - Training: Feeds model's own previous prediction (pred_{t-1}) as input.
    - Target:   Learns to predict the next intermediate step (x_t).
    - Inference: Fully autoregressive, using pred_{t-1} as input_t.
    
    This model uses `TinyRecursiveReasoningModel_ACTV5Carry`.
    It is compatible with `ACTLossHeadV4` as the loss head
    correctly uses `intermediate_labels_path` to get the target x_t.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        num_steps = self.config.halt_max_steps + 1
    
        # Vectorized generation of the full path for the initial batch
        all_paths = batch_corrupt_linear_path_v2(
            batch["inputs"].to(dtype=batch["labels"].dtype),
            batch["labels"],
            num_steps=num_steps
        )
    
        device = batch["inputs"].device
        return TinyRecursiveReasoningModel_ACTV5Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            intermediate_labels_path=all_paths,                  # [B, S, L]
            prev_prediction=torch.empty_like(batch["inputs"])    # will be filled after first step
        )

        
    def forward(self,carry: TinyRecursiveReasoningModel_ACTV5Carry, batch: Dict[str, torch.Tensor]):
        # --- Recompute paths only for sequences that were just replaced (halted),
        #     but do it in a compile‑friendly way (no Python loops).
        num_steps = self.config.halt_max_steps + 1
        all_paths_now = batch_corrupt_linear_path_v2(
            batch["inputs"].to(dtype=batch["labels"].dtype),
            batch["labels"],
            num_steps
        )  # [B, S, L]
    
        hmask_bsl = carry.halted.view(-1, 1, 1)  # [B,1,1]
        new_intermediate_labels_path = torch.where(
            hmask_bsl, all_paths_now, carry.intermediate_labels_path
        )  # [B, S, L]
    
        # --- Update inner carry and "steps" counter for sequences that are continuing
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
    
        # --- V5 autoregressive input:
        #     If a sequence was just inserted (halted==True), use x0 from the new batch.
        #     Otherwise, use the model's previous prediction.
        new_input = torch.where(
            carry.halted.view(-1, 1),
            batch["inputs"],         # x0
            carry.prev_prediction    # pred_{t-1}
        )
        # Sanitize ignored tokens
        new_input = new_input.masked_fill(new_input == IGNORE_LABEL_ID, 0)
    
        # --- Build new_current_data for the inner model
        new_current_data: Dict[str, torch.Tensor] = {}
        for k, v in carry.current_data.items():
            if k == "inputs":
                new_current_data[k] = new_input
            else:
                # Load from the fresh batch if the sequence was just inserted;
                # otherwise keep prior values to continue the rollouts.
                new_current_data[k] = torch.where(
                    carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                    batch[k],
                    v
                )
    
        # --- Time embedding indices (start at 1 on new items)
        with torch.no_grad():
            timestep_idxs = torch.where(
                carry.halted,                  # just inserted
                torch.ones_like(carry.steps),  # t = 1
                carry.steps + 1                # otherwise increment
            )
            max_steps = getattr(self.config, "halt_max_steps", getattr(self.config, "N_supervision", 16))
            timestep_idxs = timestep_idxs.clamp(min=0, max=max_steps)
    
        # --- Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry,
            new_current_data,
            timestep_idxs=timestep_idxs
        )
    
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
    
        # --- Prepare next-step autoregressive input and halting
        with torch.no_grad():
            current_prediction = torch.argmax(logits, dim=-1).to(batch["inputs"].dtype)
    
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step  # run exactly halt_max_steps steps
    
        return TinyRecursiveReasoningModel_ACTV5Carry(
            new_inner_carry,
            new_steps,
            halted,
            new_current_data,
            new_intermediate_labels_path,
            current_prediction
        ), outputs
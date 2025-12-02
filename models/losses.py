from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100

def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class ACTLossHeadv2(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # 1. Forward pass of the model
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
    
        # 2. Calculate predictions and add to outputs to fix KeyError
        # This is done inside a no_grad() context as these are for metrics/loss targets, not for differentiation.
        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            # --- FIX for KeyError ---
            outputs["preds"] = preds
            # -----------------------
    
            # 3. Prepare masks, correctness, and divisors from your original function
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Kept original variable name
    
            is_correct = mask & (preds == labels)
            seq_is_correct = (is_correct.sum(-1) == loss_counts)
    
        # 4. Calculate losses using original logic
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
    
        q_continue_loss = torch.tensor(0.0, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
        
        # 5. Calculate Metrics with the requested TRAIN vs. EVAL split
        metrics = {}
        if self.model.training:
            # TRAINING: Aggregate metrics over *all items* in the batch
            B = labels.shape[0]
            metrics = {
                # Total items in the batch
                "count": torch.tensor(B, device=labels.device, dtype=torch.float32),
                
                # Accuracy metric using your original per-sequence calculation, summed over the batch
                "accuracy": (is_correct.to(torch.float32).sum(-1) / loss_counts.clamp_min(1)).sum(),
                
                # Exact accuracy over the whole batch
                "exact_accuracy": seq_is_correct.sum(),
                
                # Q-halt accuracy over the whole batch
                "q_halt_accuracy": ((outputs["q_halt_logits"] >= 0) == seq_is_correct).sum(),
                
                # Steps over the whole batch
                "steps": new_carry.steps.sum(),
            }
        else:
            # EVALUATION: Use your original logic, counting only *halted* items
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32).sum(-1) / loss_counts.clamp_min(1)), 0.0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
    
        # 6. Update metrics with detached loss values for logging
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        if "target_q_continue" in outputs:
            metrics["q_continue_loss"] = q_continue_loss.detach()
    
        # 7. Filter outputs and return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
    
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class ACTLossHeadV3(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Forward pass to get logits and carry
        new_carry, outputs = self.model(**model_kwargs)
    
        # FINAL labels are used for metrics (unchanged behavior)
        labels = new_carry.current_data["labels"]
    
        # ---- Metrics prep (no grad) ----
        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds  # keep preds in outputs for any downstream consumers
    
            mask = (labels != IGNORE_LABEL_ID)                       # [B, T]
            loss_counts = mask.sum(-1)                               # [B]
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)    # [B, 1]
    
            is_correct = mask & (preds == labels)                    # [B, T]
            seq_is_correct = (is_correct.sum(-1) == loss_counts)     # [B]
    
            valid_metrics = new_carry.halted & (loss_counts > 0)     # [B]
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, torch.zeros_like(new_carry.steps)).sum(),
            }
    
        # ---- Losses (safe per-step supervision) ----
        logits = outputs["logits"]                                   # [B, T, V]
        B, T, V = logits.size(0), logits.size(1), logits.size(-1)
    
        # intermediate_labels_path: [B, S, T] (S = num supervision steps per example)
        intermediate = new_carry.intermediate_labels_path
        num_steps = intermediate.size(1)
    
        # new_carry.steps is 1-indexed; convert to 0-indexed and CLAMP to [0, S-1]
        step_idx = (new_carry.steps - 1).clamp(0, num_steps - 1).to(torch.long)  # [B]
        step_idx = step_idx.view(-1, 1, 1).expand(-1, 1, T)                      # [B, 1, T]
    
        # Gather the per-item labels for the current ACT step => [B, T]
        current_step_labels = torch.gather(intermediate, 1, step_idx).squeeze(1)
    
        # Sanitize class IDs to avoid device-side asserts:
        # anything not in [0, V-1] becomes IGNORE_LABEL_ID
        invalid = (current_step_labels < 0) | (current_step_labels >= V)
        safe_step_labels = current_step_labels.masked_fill(invalid, IGNORE_LABEL_ID)
    
        # Build a per-step valid mask and normalizer
        step_mask = (safe_step_labels != IGNORE_LABEL_ID)            # [B, T]
        step_loss_counts = step_mask.sum(-1).clamp_min(1)            # [B]
        step_loss_divisor = step_loss_counts.unsqueeze(-1)           # [B, 1]
    
        # Compute LM loss against the *intermediate* labels for this step.
        # Support both loss fns that accept valid_mask and those that don't.
        try:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=step_mask,
            )  # expected shape [B, T]
        except TypeError:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
            )
            per_token_loss = per_token_loss * step_mask
    
        lm_loss = (per_token_loss / step_loss_divisor).sum()
    
        # Q-halt loss (sequence-level)
        q_halt_target = seq_is_correct.to(outputs["q_halt_logits"].dtype)
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], q_halt_target, reduction="sum"
        )
    
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
    
        # Optional Q-continue loss (bootstrapped)
        q_continue_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()
    
        # Select only the requested outputs to detach/return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
    
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class ACTLossHeadV4(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    # def forward(
    #     self,
    #     return_keys: Sequence[str],
    #     # Model args
    #     **model_kwargs,
    # ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
    #     # Forward pass to get logits and carry
    #     new_carry, outputs = self.model(**model_kwargs)
    
    #     # FINAL labels are used for metrics (unchanged behavior)
    #     labels = new_carry.current_data["labels"]
    
    #     # ---- Metrics prep (no grad) ----
    #     # These metrics are for LOGGING and still compare against the FINAL label
    #     with torch.no_grad():
    #         preds = torch.argmax(outputs["logits"], dim=-1)
    #         outputs["preds"] = preds  # keep preds in outputs for any downstream consumers
    
    #         mask = (labels != IGNORE_LABEL_ID)                       # [B, T]
    #         loss_counts = mask.sum(-1)                               # [B]
    #         loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)    # [B, 1]
    
    #         is_correct = mask & (preds == labels)                    # [B, T]
    #         seq_is_correct_final = (is_correct.sum(-1) == loss_counts)     # [B] (based on FINAL label)
    
    #         valid_metrics = new_carry.halted & (loss_counts > 0)     # [B]
    #         metrics: Dict[str, torch.Tensor] = {
    #             "count": valid_metrics.sum(),
    #             "accuracy": torch.where(
    #                 valid_metrics,
    #                 (is_correct.to(torch.float32) / loss_divisor).sum(-1),
    #                 torch.zeros_like(loss_counts, dtype=torch.float32),
    #             ).sum(),
    #             "exact_accuracy": (valid_metrics & seq_is_correct_final).sum(),
    #             "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct_final)).sum(),
    #             "steps": torch.where(valid_metrics, new_carry.steps, torch.zeros_like(new_carry.steps)).sum(),
    #         }
    
    #     # ---- Losses (safe per-step supervision) ----
    #     logits = outputs["logits"]                                   # [B, T, V]
    #     B, T, V = logits.size(0), logits.size(1), logits.size(-1)
    
    #     # intermediate_labels_path: [B, S, T] (S = num supervision steps per example)
    #     intermediate = new_carry.intermediate_labels_path
    #     num_steps = intermediate.size(1)
    
    #     # new_carry.steps is 1-indexed; convert to 0-indexed and CLAMP to [0, S-1]
    #     step_idx = (new_carry.steps - 1).clamp(0, num_steps - 1).to(torch.long)  # [B]
    #     step_idx = step_idx.view(-1, 1, 1).expand(-1, 1, T)                      # [B, 1, T]
    
    #     # Gather the per-item labels for the current ACT step => [B, T]
    #     current_step_labels = torch.gather(intermediate, 1, step_idx).squeeze(1)
    
    #     # Sanitize class IDs to avoid device-side asserts:
    #     # anything not in [0, V-1] becomes IGNORE_LABEL_ID
    #     invalid = (current_step_labels < 0) | (current_step_labels >= V)
    #     safe_step_labels = current_step_labels.masked_fill(invalid, IGNORE_LABEL_ID)
    
    #     # Build a per-step valid mask and normalizer
    #     step_mask = (safe_step_labels != IGNORE_LABEL_ID)            # [B, T]
    #     step_loss_counts = step_mask.sum(-1).clamp_min(1)            # [B]
    #     step_loss_divisor = step_loss_counts.unsqueeze(-1)           # [B, 1]
    
    #     # Compute LM loss against the *intermediate* labels for this step.
    #     try:
    #         per_token_loss = self.loss_fn(
    #             logits,
    #             safe_step_labels,
    #             ignore_index=IGNORE_LABEL_ID,
    #             valid_mask=step_mask,
    #         )  # expected shape [B, T]
    #     except TypeError:
    #         per_token_loss = self.loss_fn(
    #             logits,
    #             safe_step_labels,
    #             ignore_index=IGNORE_LABEL_ID,
    #         )
    #         per_token_loss = per_token_loss * step_mask
    
    #     lm_loss = (per_token_loss / step_loss_divisor).sum()
    
    #     # --- FIX for Issue 2: Align Halt Target with Guidance Target ---
    #     # The Q-halt loss must target correctness against the INTERMEDIATE label,
    #     # not the FINAL label, to align with the lm_loss.
    #     with torch.no_grad(): # No gradients needed for creating a target
    #         step_is_correct = step_mask & (preds == safe_step_labels)
    #         step_loss_counts_metric = step_mask.sum(-1) # count for comparison
    #         # This is the new, aligned halt target:
    #         step_seq_is_correct = (step_is_correct.sum(-1) == step_loss_counts_metric)
        
    #     q_halt_target = step_seq_is_correct.to(outputs["q_halt_logits"].dtype)
    #     # --- End Fix ---

    #     q_halt_loss = F.binary_cross_entropy_with_logits(
    #         outputs["q_halt_logits"], q_halt_target, reduction="sum"
    #     )
    
    #     metrics.update({
    #         "lm_loss": lm_loss.detach(),
    #         "q_halt_loss": q_halt_loss.detach(),
    #     })
    
    #     # Optional Q-continue loss (bootstrapped)
    #     q_continue_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    #     if "target_q_continue" in outputs:
    #         q_continue_loss = F.binary_cross_entropy_with_logits(
    #             outputs["q_continue_logits"],
    #             outputs["target_q_continue"],
    #             reduction="sum",
    #         )
    #         metrics["q_continue_loss"] = q_continue_loss.detach()
    
    #     # Select only the requested outputs to detach/return
    #     detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
    
    #     total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
    #     return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Forward pass to get logits and carry
        new_carry, outputs = self.model(**model_kwargs)
    
        # FINAL labels are used for metrics (unchanged behavior)
        labels = new_carry.current_data["labels"]
    
        # ---- Metrics prep (no grad) ----
        # These metrics are for LOGGING and still compare against the FINAL label
        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds  # keep preds in outputs
    
            mask = (labels != IGNORE_LABEL_ID)                       # [B, T]
            loss_counts = mask.sum(-1)                               # [B]
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)    # [B, 1]
    
            is_correct = mask & (preds == labels)                    # [B, T]
            seq_is_correct_final = (is_correct.sum(-1) == loss_counts)     # [B] (based on FINAL label)
    
            # Since we run fixed steps, check all items that have finished
            valid_metrics = new_carry.halted & (loss_counts > 0)     # [B]
            
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct_final).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, torch.zeros_like(new_carry.steps)).sum(),
            }
    
        # ---- Losses (safe per-step supervision) ----
        logits = outputs["logits"]                                   # [B, T, V]
        B, T, V = logits.size(0), logits.size(1), logits.size(-1)
    
        # intermediate_labels_path: [B, S, T] (S = num supervision steps per example)
        intermediate = new_carry.intermediate_labels_path
        num_steps = intermediate.size(1)
    
        # new_carry.steps is 1-indexed; convert to 0-indexed and CLAMP to [0, S-1]
        #step_idx = (new_carry.steps - 1).clamp(0, num_steps - 1).to(torch.long)  # [B]
        step_idx = new_carry.steps.clamp(0, num_steps - 1).to(torch.long)  # [B]
        step_idx = step_idx.view(-1, 1, 1).expand(-1, 1, T)                      # [B, 1, T]
        current_step_labels = torch.gather(intermediate, 1, step_idx).squeeze(1) # Gather the per-item labels for the current ACT step => [B, T]

        step_idx = new_carry.steps.clamp(0, num_steps - 1).to(torch.long)  # [B]
    
        # Sanitize class IDs
        invalid = (current_step_labels < 0) | (current_step_labels >= V)
        safe_step_labels = current_step_labels.masked_fill(invalid, IGNORE_LABEL_ID)
    
        # Build a per-step valid mask and normalizer
        step_mask = (safe_step_labels != IGNORE_LABEL_ID)            # [B, T]
        step_loss_counts = step_mask.sum(-1).clamp_min(1)            # [B]
        step_loss_divisor = step_loss_counts.unsqueeze(-1)           # [B, 1]
    
        # Compute LM loss against the *intermediate* labels for this step.
        try:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=step_mask,
            )
        except TypeError:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
            )
            per_token_loss = per_token_loss * step_mask
    
        lm_loss = (per_token_loss / step_loss_divisor).sum()
    
        # --- START: MODIFICATION ---
        # Total loss is ONLY the lm_loss. All Q-losses are removed.
        total_loss = lm_loss
    
        metrics.update({
            "lm_loss": lm_loss.detach(),
            # q_halt_loss and q_continue_loss are removed
        })
        # --- END: MODIFICATION ---

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
    
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class ACTLossHeadV5(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs) 

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Forward pass to get logits and carry
        new_carry, outputs = self.model(**model_kwargs)
    
        # FINAL labels are used for metrics (unchanged behavior)
        labels = new_carry.current_data["labels"]
    
        # ---- Metrics prep (no grad) ----
        # These metrics are for LOGGING and still compare against the FINAL label
        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds  # keep preds in outputs

            # -------- ONE-SHOT METRIC SLICE --------
            # If the sequence is one-shot [train1 | train2 | test], compute metrics on TEST third only.
            B, T = labels.shape[0], labels.shape[1]
            metric_slice = slice(0, T)
            if T % 3 == 0 and T >= 900:
                third = T // 3
                metric_slice = slice(2 * third, 3 * third)

            labels_m = labels[:, metric_slice]
            preds_m  = preds[:,  metric_slice]

            mask = (labels_m != IGNORE_LABEL_ID)                    # [B, Tm]
            loss_counts = mask.sum(-1)                              # [B]
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)   # [B, 1]
    
            is_correct = mask & (preds_m == labels_m)               # [B, Tm]
            seq_is_correct_final = (is_correct.sum(-1) == loss_counts)  # [B]
    
            # Since we run fixed steps, check all items that have finished
            valid_metrics = new_carry.halted & (loss_counts > 0)    # [B]
            
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct_final).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, torch.zeros_like(new_carry.steps)).sum(),
            }
    
        # ---- Losses (safe per-step supervision) ----
        logits = outputs["logits"]                                   # [B, T, V]
        B, T, V = logits.size(0), logits.size(1), logits.size(-1)
    
        # intermediate_labels_path: [B, S, T] (S = num supervision steps per example)
        intermediate = new_carry.intermediate_labels_path
        num_steps = intermediate.size(1)
    
        # new_carry.steps is treated as 0-indexed in your current code; clamp to [0, S-1]
        step_idx = new_carry.steps.clamp(0, num_steps - 1).to(torch.long)   # [B]
        step_idx = step_idx.view(-1, 1, 1).expand(-1, 1, T)                  # [B, 1, T]
        current_step_labels = torch.gather(intermediate, 1, step_idx).squeeze(1)  # [B, T]

        # (Kept for parity with your reference; not reused below)
        step_idx = new_carry.steps.clamp(0, num_steps - 1).to(torch.long)   # [B]
    
        # Sanitize class IDs
        invalid = (current_step_labels < 0) | (current_step_labels >= V)
        safe_step_labels = current_step_labels.masked_fill(invalid, IGNORE_LABEL_ID)
    
        # Build a per-step valid mask and normalizer
        step_mask = (safe_step_labels != IGNORE_LABEL_ID)            # [B, T]
        step_loss_counts = step_mask.sum(-1).clamp_min(1)            # [B]
        step_loss_divisor = step_loss_counts.unsqueeze(-1)           # [B, 1]
    
        # Compute LM loss against the *intermediate* labels for this step.
        try:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=step_mask,
            )
        except TypeError:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
            )
            per_token_loss = per_token_loss * step_mask
    
        lm_loss = (per_token_loss / step_loss_divisor).sum()
    
        # --- START: MODIFICATION ---
        # Total loss is ONLY the lm_loss. All Q-losses are removed (same as your V4).
        total_loss = lm_loss
    
        metrics.update({
            "lm_loss": lm_loss.detach(),
            # q_halt_loss and q_continue_loss are removed
        })
        # --- END: MODIFICATION ---

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
    
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()




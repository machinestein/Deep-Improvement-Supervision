"""
N-Queens Evaluator for Tiny Recursive Models (TRM)

This evaluator computes accuracy metrics for N-Queens puzzle solving.
It checks if predicted solutions are valid (no conflicts) and match the ground truth.

Metrics computed:
- Exact match accuracy: Percentage of puzzles where prediction exactly matches label
- Valid solution rate: Percentage of predictions that are valid N-Queens solutions
- Constraint satisfaction: Average percentage of satisfied constraints per puzzle
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata


class NQueens:
    """
    N-Queens evaluator for TRM
    
    This evaluator tracks predictions during evaluation and computes:
    1. Exact match accuracy (prediction == label)
    2. Valid solution rate (prediction is a valid N-Queens solution)
    3. Constraint satisfaction rate (percentage of constraints satisfied)
    """
    
    # Specify which outputs from the model we need
    required_outputs = {"preds", "labels"}
    
    def __init__(self, 
                 data_path: str, 
                 eval_metadata: PuzzleDatasetMetadata,
                 board_size: int = 8):
        """
        Initialize N-Queens evaluator
        
        Args:
            data_path: Path to the dataset (not used but required by TRM interface)
            eval_metadata: Metadata about the dataset
            board_size: Size of the N-Queens board (N in N-Queens)
        """
        super().__init__()
        self.board_size = board_size
        self.blank_identifier_id = eval_metadata.blank_identifier_id
        
        # Local statistics (accumulated during evaluation)
        self._local_stats = {
            'total': 0,
            'exact_match': 0,
            'valid_solutions': 0,
            'total_constraints': 0,
            'satisfied_constraints': 0
        }
    
    def begin_eval(self):
        """
        Called at the start of each evaluation run
        Reset all statistics
        """
        self._local_stats = {
            'total': 0,
            'exact_match': 0,
            'valid_solutions': 0,
            'total_constraints': 0,
            'satisfied_constraints': 0
        }
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Process a batch of predictions
        
        Args:
            batch: Dictionary containing batch data (includes 'labels')
            preds: Dictionary containing model predictions (includes 'preds')
        """
        # Move tensors to CPU for processing
        predictions = preds["preds"].cpu().numpy()  # Shape: (batch_size, seq_len)
        labels = batch["labels"].cpu().numpy()      # Shape: (batch_size, seq_len)
        
        # Get puzzle identifiers to filter out padding
        if "puzzle_identifiers" in batch:
            puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
            # Remove padding (where puzzle_identifier == blank_identifier_id)
            mask = puzzle_ids != self.blank_identifier_id
            predictions = predictions[mask]
            labels = labels[mask]
        
        # Process each puzzle in the batch
        for pred_flat, label_flat in zip(predictions, labels):
            # Decode: vocabulary is 0 (PAD), 1 (empty), 2 (queen)
            # Subtract 1 to get: -1 (PAD), 0 (empty), 1 (queen)
            # Then clip to get: 0 (empty), 1 (queen)
            pred_board = np.clip(pred_flat - 1, 0, 1).reshape(self.board_size, self.board_size)
            label_board = np.clip(label_flat - 1, 0, 1).reshape(self.board_size, self.board_size)
            
            # Update statistics
            self._local_stats['total'] += 1
            
            # 1. Check exact match
            if np.array_equal(pred_board, label_board):
                self._local_stats['exact_match'] += 1
            
            # 2. Check if prediction is a valid solution
            is_valid, num_satisfied, num_total = self._check_solution_validity(pred_board)
            if is_valid:
                self._local_stats['valid_solutions'] += 1
            
            # 3. Track constraint satisfaction
            self._local_stats['satisfied_constraints'] += num_satisfied
            self._local_stats['total_constraints'] += num_total
    
    def _check_solution_validity(self, board: np.ndarray):
        """
        Check if a board represents a valid N-Queens solution
        
        Args:
            board: NxN numpy array (0=empty, 1=queen)
        
        Returns:
            (is_valid, num_satisfied_constraints, num_total_constraints)
        """
        N = self.board_size
        num_satisfied = 0
        num_total = 0
        
        # Check 1: Exactly N queens should be placed
        num_queens = np.sum(board)
        num_total += 1
        if num_queens == N:
            num_satisfied += 1
        
        # Check 2: Each row should have exactly 1 queen
        for i in range(N):
            num_total += 1
            if np.sum(board[i, :]) == 1:
                num_satisfied += 1
        
        # Check 3: Each column should have exactly 1 queen
        for j in range(N):
            num_total += 1
            if np.sum(board[:, j]) == 1:
                num_satisfied += 1
        
        # Check 4: Each main diagonal should have at most 1 queen
        for d in range(-(N-1), N):
            num_total += 1
            if np.trace(board, offset=d) <= 1:
                num_satisfied += 1
        
        # Check 5: Each anti-diagonal should have at most 1 queen
        flipped = np.fliplr(board)
        for d in range(-(N-1), N):
            num_total += 1
            if np.trace(flipped, offset=d) <= 1:
                num_satisfied += 1
        
        # Solution is valid if all constraints are satisfied
        is_valid = (num_satisfied == num_total)
        
        return is_valid, num_satisfied, num_total
    
    def result(self, 
               save_path: Optional[str], 
               rank: int, 
               world_size: int, 
               group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        """
        Compute final evaluation metrics
        
        This is called after all batches have been processed.
        Aggregates statistics across all processes (if using multi-GPU).
        
        Args:
            save_path: Path to save results (not used for N-Queens)
            rank: Process rank (0 for main process)
            world_size: Total number of processes
            group: Process group for distributed training
        
        Returns:
            Dictionary of metrics (only on rank 0, None on other ranks)
        """
        # Convert local stats to tensor for distributed reduction
        stats_tensor = torch.tensor([
            self._local_stats['total'],
            self._local_stats['exact_match'],
            self._local_stats['valid_solutions'],
            self._local_stats['satisfied_constraints'],
            self._local_stats['total_constraints']
        ], dtype=torch.float32, device='cuda')
        
        # Gather statistics from all processes to rank 0
        if world_size > 1:
            dist.reduce(stats_tensor, dst=0, group=group)
        
        # Only rank 0 computes and returns metrics
        if rank != 0:
            return None
        
        # Extract aggregated statistics
        total = int(stats_tensor[0].item())
        exact_match = int(stats_tensor[1].item())
        valid_solutions = int(stats_tensor[2].item())
        satisfied_constraints = int(stats_tensor[3].item())
        total_constraints = int(stats_tensor[4].item())
        
        # Compute metrics
        if total == 0:
            return {
                'NQueens/exact_match_accuracy': 0.0,
                'NQueens/valid_solution_rate': 0.0,
                'NQueens/constraint_satisfaction': 0.0
            }
        
        exact_match_accuracy = exact_match / total
        valid_solution_rate = valid_solutions / total
        constraint_satisfaction = satisfied_constraints / total_constraints if total_constraints > 0 else 0.0
        
        # Print detailed results
        print("\n" + "="*80)
        print("N-Queens Evaluation Results")
        print("="*80)
        print(f"Total puzzles evaluated: {total}")
        print(f"Exact matches: {exact_match} ({exact_match_accuracy*100:.2f}%)")
        print(f"Valid solutions: {valid_solutions} ({valid_solution_rate*100:.2f}%)")
        print(f"Constraint satisfaction: {satisfied_constraints}/{total_constraints} ({constraint_satisfaction*100:.2f}%)")
        print("="*80 + "\n")
        
        return {
            'NQueens/exact_match_accuracy': exact_match_accuracy,
            'NQueens/valid_solution_rate': valid_solution_rate,
            'NQueens/constraint_satisfaction': constraint_satisfaction
        }

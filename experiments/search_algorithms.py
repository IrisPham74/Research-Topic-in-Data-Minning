"""
Search algorithms adapted for Task 1 experiments.
Wraps the original search classes to support custom model evaluators.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import pandas as pd
from config import GAConfig, BanditConfig
from prompt import Prompt
from model_evaluator import BlackBoxModelEvaluator

# Import original search algorithms
from evolutionary import EvolutionarySearch as OriginalEvolutionarySearch
from greedy import MultiStartGreedySearch as OriginalMultiStartGreedySearch
from bandit import BanditSearch as OriginalBanditSearch


class ModelAwareEvolutionarySearch(OriginalEvolutionarySearch):
    """Evolutionary search with custom model evaluator supporting multiple tasks."""
    
    def __init__(self, vocab: List[str], val_df: pd.DataFrame, test_df: pd.DataFrame,
                 cfg: GAConfig, model_type: str = "gpt2", model_id: str = None,
                 task_type: str = "sst2", label_list: List[str] = None):
        # Don't call super().__init__ yet
        self.vocab = vocab
        self.val = val_df
        self.test = test_df
        self.cfg = cfg
        
        # Use custom evaluator with task support
        self.evaluator = BlackBoxModelEvaluator(model_type, model_id, task_type, label_list)
        
        # Initialize random seed
        import random
        random.seed(cfg.seed)


class ModelAwareMultiStartGreedySearch(OriginalMultiStartGreedySearch):
    """Multi-start greedy search with custom model evaluator supporting multiple tasks."""
    
    def __init__(self, vocab: List[str], val_df: pd.DataFrame, test_df: pd.DataFrame,
                 cfg: GAConfig, model_type: str = "gpt2", model_id: str = None,
                 restarts: int = 5, task_type: str = "sst2", label_list: List[str] = None):
        # Initialize base attributes
        self.vocab = vocab
        self.val = val_df
        self.test = test_df
        self.cfg = cfg
        self.restarts = restarts
        
        # Use custom evaluator with task support
        self.evaluator = BlackBoxModelEvaluator(model_type, model_id, task_type, label_list)
        
        # Initialize random seed
        import random
        random.seed(cfg.seed)


class ModelAwareBanditSearch(OriginalBanditSearch):
    """Bandit search with custom model evaluator supporting multiple tasks."""
    
    def __init__(self, vocab: List[str], val_df: pd.DataFrame, test_df: pd.DataFrame,
                 cfg: BanditConfig, model_type: str = "gpt2", model_id: str = None,
                 task_type: str = "sst2", label_list: List[str] = None):
        # Initialize base attributes
        self.vocab = vocab
        self.val = val_df
        self.test = test_df
        self.cfg = cfg
        self.num_triggers = cfg.num_trigger
        self.iters = cfg.iters
        self.epsilon = cfg.epsilon
        
        # Use custom evaluator with task support
        self.evaluator = BlackBoxModelEvaluator(model_type, model_id, task_type, label_list)
        
        # Initialize random seed and bandit-specific attributes
        import random
        random.seed(cfg.seed)
        
        # Initialize token statistics
        self.token_counts = {token: 0 for token in vocab}
        self.token_rewards = {token: 0.0 for token in vocab}


from dataclasses import dataclass
from typing import Optional

@dataclass
class GAConfig:
    """Configuration for genetic algorithm based search."""
    pop: int = 40
    elite: int = 4
    gens: int = 20
    max_tokens: int = 16
    mut_p: float = 0.5
    ins_p: float = 0.25
    del_p: float = 0.25
    swap_p: float = 0.25
    cx_p: float = 0.8
    tourn_k: int = 4
    seed: int = 123
    early_patience: int = 5
    num_trigger: Optional[int] = None
    search_algo: str = "evo"

@dataclass
class BanditConfig(GAConfig):
    """Configuration for bandit search."""
    epsilon: float = 0.2
    iters: int = 200
    search_algo: str = "bandit"
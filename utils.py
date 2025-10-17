import os
import re
import random
import pandas as pd
from typing import List, Tuple


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)


def load_dataset_csvs(directory: str = "small_dataset") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    return (
        pd.read_csv(os.path.join(directory, "train.csv")),
        pd.read_csv(os.path.join(directory, "val.csv")),
        pd.read_csv(os.path.join(directory, "test.csv")),
    )


def load_vocab(path: str = "Vcand/vcand.txt") -> List[str]:
    """Load vocabulary from file."""
    with open(path, "r", encoding="utf-8") as f:
        vocab = [word.strip() for word in f if word.strip()]

    # Filter and deduplicate
    vocab = list(dict.fromkeys([
        word for word in vocab
        if re.fullmatch(r"[A-Za-z\-']{3,}", word)
    ]))
    return vocab


def create_searcher(cfg, vocab, val_df, test_df):
    """Factory function to create appropriate searcher."""
    algo = cfg.search_algo.lower()

    if algo == "evo":
        from evolutionary import EvolutionarySearch
        return EvolutionarySearch(vocab, val_df, test_df, cfg)
    elif algo == "greedy":
        from greedy import GreedySearch
        return GreedySearch(vocab, val_df, test_df, cfg)
    elif algo in ["multi", "multi_greedy", "multistart"]:
        from greedy import MultiStartGreedySearch
        return MultiStartGreedySearch(vocab, val_df, test_df, cfg, restarts=5)
    elif algo == "bandit":
        from bandit import BanditSearch
        return BanditSearch(vocab, val_df, test_df, cfg)
    else:
        raise ValueError(f"Unknown search algorithm: {cfg.search_algo}")
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from prompt import Prompt
from evaluator import BlackBoxEvaluator
from config import GAConfig

class BaseSearch(ABC):
    """Base class for all search algorithms."""

    def __init__(self, vocab: List[str], val_df: pd.DataFrame, test_df: pd.DataFrame, cfg: GAConfig):
        self.vocab = vocab
        self.val = val_df
        self.test = test_df
        self.cfg = cfg
        self.evaluator = BlackBoxEvaluator()
        random.seed(cfg.seed)

    def random_prompt(self) -> Prompt:
        """Generate a random prompt."""
        if self.cfg.num_trigger is not None:
            L = min(self.cfg.num_trigger, self.cfg.max_tokens, len(self.vocab))
        else:
            L = random.randint(max(3, self.cfg.max_tokens // 3), self.cfg.max_tokens)
            L = min(L, len(self.vocab))
        return Prompt(tokens=random.sample(self.vocab, k=L))

    def evaluate_prompt(self, prompt: Prompt, df: pd.DataFrame) -> float:
        """Evaluate a prompt on the given dataset."""
        prompt_text = prompt.text()
        texts = df["sentence"].tolist()
        gold = df["label"].tolist()
        inputs = self.evaluator.build_inputs(prompt_text, texts)
        preds = self.evaluator.predict_batch(inputs)
        return sum(int(a == b) for a, b in zip(preds, gold)) / max(1, len(gold))

    def evaluate_population(self, population: List[Prompt], df: pd.DataFrame):
        """Evaluate a population of prompts."""
        for individual in population:
            if individual.fitness is None:
                individual.fitness = self.evaluate_prompt(individual, df)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the search algorithm."""
        pass
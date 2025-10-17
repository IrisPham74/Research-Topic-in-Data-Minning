import random
from typing import List, Dict, Any
from prompt import Prompt
from config import BanditConfig
from base_search import BaseSearch
import pandas as pd


class BanditSearch(BaseSearch):
    """Multi-armed bandit search for prompt optimization."""

    def __init__(self, vocab: List[str], val_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BanditConfig):
        super().__init__(vocab, val_df, test_df, cfg)
        self.num_triggers = cfg.num_trigger
        self.epsilon = cfg.epsilon
        self.iters = cfg.iters

    def run(self) -> Dict[str, Any]:
        """Run the bandit search."""
        # Initialize bandit values and counts
        bandit_values = [{token: 0.0 for token in self.vocab}
                         for _ in range(self.num_triggers)]
        counts = [{token: 1 for token in self.vocab}
                  for _ in range(self.num_triggers)]

        # Initialize current and best prompts
        current_tokens = random.choices(self.vocab, k=self.num_triggers)
        best_prompt = Prompt(tokens=current_tokens[:])
        best_reward = self.evaluate_prompt(best_prompt, self.val)

        history = [{"iter": 0, "best": best_reward, "text": best_prompt.text()}]

        for iteration in range(1, self.iters + 1):
            # Select position to update
            position = random.randint(0, self.num_triggers - 1)

            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                candidate = random.choice(self.vocab)
            else:
                candidate = max(bandit_values[position],
                                key=lambda t: bandit_values[position][t])

            # Create trial prompt
            trial_tokens = current_tokens[:]
            trial_tokens[position] = candidate
            trial_prompt = Prompt(tokens=trial_tokens)
            reward = self.evaluate_prompt(trial_prompt, self.val)

            # Update bandit estimates
            counts[position][candidate] += 1
            bandit_values[position][candidate] += (
                                                          reward - bandit_values[position][candidate]
                                                  ) / counts[position][candidate]

            # Update best solution
            if reward > best_reward:
                best_reward = reward
                best_prompt = trial_prompt.copy()

            # Move to new selection
            current_tokens[position] = candidate

            # Record history periodically
            if iteration % 5 == 0:
                history.append({
                    "iter": iteration,
                    "best": best_reward,
                    "text": best_prompt.text()
                })
                print(f"[{iteration}] Reward={reward:.4f} | Best={best_reward:.4f}")

        # Final test evaluation
        test_acc = self.evaluate_prompt(best_prompt, self.test)
        print(f"[Bandit Final] Val Acc={best_reward:.4f}, Test Acc={test_acc:.4f}")

        return {
            "best_prompt": best_prompt.text(),
            "val_acc": best_reward,
            "test_acc": test_acc,
            "history": history
        }
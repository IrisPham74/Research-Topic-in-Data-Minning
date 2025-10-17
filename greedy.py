import random
from typing import List, Dict, Any
from prompt import Prompt
from config import GAConfig
from base_search import BaseSearch
import pandas as pd

class GreedySearch(BaseSearch):
    """Greedy hill-climbing search for prompt optimization."""

    def mutate_prompt(self, prompt: Prompt) -> Prompt:
        """Generate a mutated version of the prompt."""
        t = prompt.tokens[:]
        if random.random() < self.cfg.ins_p and len(t) < self.cfg.max_tokens:
            t.insert(random.randrange(len(t) + 1), random.choice(self.vocab))
        if len(t) > 3 and random.random() < self.cfg.del_p:
            t.pop(random.randrange(len(t)))
        if len(t) > 3 and random.random() < self.cfg.swap_p:
            i, j = random.sample(range(len(t)), 2)
            t[i], t[j] = t[j], t[i]
        if len(t) > 0 and random.random() < self.cfg.mut_p:
            t[random.randrange(len(t))] = random.choice(self.vocab)
        return Prompt(tokens=t[:self.cfg.max_tokens])

    def run(self) -> Dict[str, Any]:
        """Run the greedy search."""
        current = self.random_prompt()
        current.fitness = self.evaluate_prompt(current, self.val)
        best = current.copy()

        history = [{"step": 0, "best": best.fitness, "text": best.text()}]
        no_improvement = 0

        print(f"[Greedy Init] Fitness = {best.fitness:.4f}")

        for step in range(1, self.cfg.gens + 1):
            # Generate and evaluate neighbors
            neighbors = [self.mutate_prompt(current) for _ in range(self.cfg.pop)]
            self.evaluate_population(neighbors, self.val)
            neighbors.sort(key=lambda x: x.fitness, reverse=True)
            best_neighbor = neighbors[0]

            # Update current solution
            if best_neighbor.fitness > current.fitness:
                current = best_neighbor
                if best_neighbor.fitness > best.fitness:
                    best = best_neighbor.copy()
                    no_improvement = 0
                    print(f"[Greedy Step {step}] Improved to {best.fitness:.4f}")
                else:
                    no_improvement += 1
            else:
                no_improvement += 1

            history.append({"step": step, "best": best.fitness, "text": best.text()})

            # Early stopping
            if no_improvement >= self.cfg.early_patience:
                print(f"[Greedy Stop] No improvement for {self.cfg.early_patience} steps.")
                break

        # Final test evaluation
        test_acc = self.evaluate_prompt(best, self.test)
        print(f"[Greedy Final] Val Acc={best.fitness:.4f}, Test Acc={test_acc:.4f}")

        return {
            "best_prompt": best.text(),
            "val_acc": best.fitness,
            "test_acc": test_acc,
            "history": history,
        }


class MultiStartGreedySearch(GreedySearch):
    """Multi-start greedy search with multiple restarts."""

    def __init__(self, vocab: List[str], val_df: pd.DataFrame, test_df: pd.DataFrame,
                 cfg: GAConfig, restarts: int = 5):
        super().__init__(vocab, val_df, test_df, cfg)
        self.restarts = restarts

    def run(self) -> Dict[str, Any]:
        """Run multi-start greedy search."""
        print(f"\n===== Multi-Start Greedy Search: {self.restarts} runs =====")
        all_results = []
        global_best = None

        for run in range(1, self.restarts + 1):
            print(f"\n--- Restart {run}/{self.restarts} ---")

            # Run single greedy search
            current = self.random_prompt()
            current.fitness = self.evaluate_prompt(current, self.val)
            best_local = current.copy()
            no_improvement = 0

            for step in range(1, self.cfg.gens + 1):
                neighbors = [self.mutate_prompt(current) for _ in range(self.cfg.pop)]
                self.evaluate_population(neighbors, self.val)
                neighbors.sort(key=lambda x: x.fitness, reverse=True)
                best_neighbor = neighbors[0]

                if best_neighbor.fitness > current.fitness:
                    current = best_neighbor
                    if best_neighbor.fitness > best_local.fitness:
                        best_local = best_neighbor.copy()
                        no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= self.cfg.early_patience:
                    break

            test_acc = self.evaluate_prompt(best_local, self.test)
            result = {
                "best_prompt": best_local.text(),
                "val_acc": best_local.fitness,
                "test_acc": test_acc,
            }
            all_results.append(result)

            if global_best is None or result["val_acc"] > global_best["val_acc"]:
                global_best = result

        print(f"\n[Multi-Start Summary] Best Val Acc={global_best['val_acc']:.4f}, "
              f"Test Acc={global_best['test_acc']:.4f}")
        return global_best
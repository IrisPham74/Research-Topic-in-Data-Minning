import random
from typing import List, Dict, Any, Tuple
from prompt import Prompt
from base_search import BaseSearch


class EvolutionarySearch(BaseSearch):
    """Evolutionary algorithm for prompt optimization."""

    def mutate(self, p: Prompt) -> Prompt:
        """Apply mutation operations to a prompt."""
        t = p.tokens[:]
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

    def crossover(self, a: Prompt, b: Prompt) -> Tuple[Prompt, Prompt]:
        """Perform crossover between two prompts."""
        if (random.random() > self.cfg.cx_p or len(a.tokens) < 2 or
                len(b.tokens) < 2):
            return Prompt(a.tokens[:]), Prompt(b.tokens[:])

        ia = random.randrange(1, len(a.tokens))
        ib = random.randrange(1, len(b.tokens))
        return (
            Prompt(a.tokens[:ia] + b.tokens[ib:]),
            Prompt(b.tokens[:ib] + a.tokens[ia:])
        )

    def tournament_selection(self, population: List[Prompt]) -> Prompt:
        """Select a prompt using tournament selection."""
        candidates = random.sample(
            population,
            k=min(self.cfg.tourn_k, len(population))
        )
        candidates.sort(key=lambda x: x.fitness or -1, reverse=True)
        return Prompt(tokens=candidates[0].tokens[:], fitness=candidates[0].fitness)

    def run(self) -> Dict[str, Any]:
        """Run the evolutionary search."""
        # Initialize population
        population = [self.random_prompt() for _ in range(self.cfg.pop)]
        self.evaluate_population(population, self.val)
        population.sort(key=lambda x: x.fitness, reverse=True)

        best = population[0].copy()
        no_improvement = 0
        history = [{"gen": 0, "best": best.fitness, "text": best.text()}]

        # Evolution loop
        for generation in range(1, self.cfg.gens + 1):
            # Elitism
            elites = population[:self.cfg.elite]
            offspring = []

            # Generate offspring
            while len(elites) + len(offspring) < self.cfg.pop:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([self.mutate(child1), self.mutate(child2)])

            # Create new population
            population = elites + offspring[:self.cfg.pop - len(elites)]
            self.evaluate_population(population, self.val)
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Update best solution
            if population[0].fitness > best.fitness:
                best = population[0].copy()
                no_improvement = 0
            else:
                no_improvement += 1

            # Record history
            history.append({
                "gen": generation,
                "best": best.fitness,
                "text": best.text()
            })

            # Early stopping
            if no_improvement >= self.cfg.early_patience:
                break

        # Final evaluation on test set
        test_acc = self.evaluate_prompt(best, self.test)

        return {
            "best_prompt": best.text(),
            "val_acc": best.fitness,
            "test_acc": test_acc,
            "history": history
        }
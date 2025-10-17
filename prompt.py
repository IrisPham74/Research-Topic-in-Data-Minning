from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Prompt:
    tokens: List[str]
    fitness: Optional[float] = None

    def text(self) -> str:
        return "Instruction: " + " ".join(self.tokens)

    def copy(self):
        return Prompt(tokens=self.tokens[:], fitness=self.fitness)
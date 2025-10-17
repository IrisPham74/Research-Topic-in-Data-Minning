import hashlib
from typing import Dict, List
from transformers import pipeline


class BlackBoxEvaluator:
    """Treat the model as a black box; return label {0,1} given full input string."""

    def __init__(self):
        # HF sentiment pipeline (POSITIVE/NEGATIVE)
        self.clf = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.cache: Dict[str, int] = {}
        self.template = (
            "You are a sentiment classifier. Decide if the review is Positive or Negative.\n"
            "Guideline: {prompt}\nReview: {text}\n"
            "Answer with exactly one word: Positive or Negative."
        )

    def _key(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def query_one(self, full_input: str) -> int:
        k = self._key(full_input)
        if k in self.cache:
            return self.cache[k]
        out = self.clf(full_input, truncation=True)[0]  # {'label':'POSITIVE'|'NEGATIVE', 'score':...}
        lbl = 1 if out["label"].lower().startswith("pos") else 0
        self.cache[k] = lbl
        return lbl

    def predict_batch(self, inputs: List[str]) -> List[int]:
        return [self.query_one(x) for x in inputs]

    def build_inputs(self, prompt_text: str, texts: List[str]) -> List[str]:
        return [self.template.format(prompt=prompt_text, text=t) for t in texts]
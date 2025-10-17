"""
Multi-Model Evaluator for Task 1
Supports GPT-2, GPT-3.5, and LLaMA as black-box classifiers
"""

import hashlib
import os
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)


class BlackBoxModelEvaluator:
    """
    Black-box evaluator that supports multiple models.
    Only uses model outputs (labels) for optimization.
    """
    
    def __init__(self, model_type: str = "gpt2", model_id: Optional[str] = None, 
                 task_type: str = "sst2", label_list: Optional[List[str]] = None):
        """
        Initialize evaluator with specified model and task.
        
        Args:
            model_type: One of 'gpt2', 'gpt3.5', 'llama'
            model_id: Specific model identifier (uses defaults if None)
            task_type: One of 'sst2', 'nli', 'sarcasm', 'topic'
            label_list: List of labels for topic classification (required for task_type='topic')
        """
        self.model_type = model_type
        self.task_type = task_type
        self.label_list = label_list
        self.cache: Dict[str, int] = {}
        
        # Template library for different tasks (Iris's format + Gemini's suggestions)
        # System role: task definition + prompt tokens + instruction
        # User role: just the input text(s)
        self.templates = {
            "sst2": "You are a sentimental analysis. {prompt} Give me Positive or Negative",
            "nli": "You are a Natural Language Inference (NLI) analysis system. {prompt} Give me Entailment, Contradiction, or Neutral",
            "sarcasm": "You are a sarcasm detection analysis. {prompt} Give me Sarcastic or Not Sarcastic",
            "topic": "You are a topic classification analysis. {prompt} Choose one topic from the following list: {label_list}. Give me only the topic name."
        }
        
        # Validate task type
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}. Must be one of {list(self.templates.keys())}")
        
        # Validate label_list for topic classification
        if task_type == "topic" and not label_list:
            raise ValueError("label_list is required for topic classification task")
        
        # Set the system template based on task
        self.system_template = self.templates[task_type]
        
        # Build simple template for models like GPT-2
        if task_type == "topic":
            label_list_str = ", ".join(label_list) if label_list else ""
            self.simple_template = f"{self.system_template.replace('{label_list}', label_list_str)}\n{{text}}"
        else:
            self.simple_template = f"{self.system_template}\n{{text}}"
        
        # Initialize model
        if model_type == "gpt2":
            self._init_gpt2(model_id or "gpt2")
        elif model_type == "gpt3.5":
            self._init_gpt35(model_id or "gpt-3.5-turbo")
        elif model_type == "llama":
            self._init_llama(model_id or "meta-llama/Llama-2-7b-chat-hf")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _init_gpt2(self, model_id: str):
        """Initialize GPT-2 model."""
        print(f"Loading GPT-2: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"[OK] GPT-2 loaded on {self.device}")
    
    def _init_gpt35(self, model_id: str):
        """Initialize GPT-3.5 via OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai.api_key = api_key
        self.openai = openai
        self.model_id = model_id
        print(f"[OK] GPT-3.5 initialized with API key")
    
    def _init_llama(self, model_id: str):
        """Initialize LLaMA model."""
        print(f"Loading LLaMA: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        print(f"[OK] LLaMA loaded")
    
    def _key(self, s: str) -> str:
        """Generate cache key for input string."""
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    
    def _parse_sentiment(self, text: str) -> int:
        """
        Parse model output to extract sentiment label for SST-2.
        Returns 1 for Positive, 0 for Negative.
        """
        text_lower = text.lower().strip()
        
        # Check for explicit labels
        if "positive" in text_lower:
            return 1
        if "negative" in text_lower:
            return 0
        
        # Check for single word responses
        if text_lower.startswith("pos"):
            return 1
        if text_lower.startswith("neg"):
            return 0
        
        # Default to negative if unclear (conservative)
        return 0
    
    def _parse_nli_label(self, text: str) -> int:
        """
        Parse model output to extract NLI label.
        Returns: 0 for Entailment, 1 for Contradiction, 2 for Neutral
        """
        text_lower = text.lower().strip()
        
        # Check for explicit labels
        if "entailment" in text_lower:
            return 0
        if "contradiction" in text_lower:
            return 1
        if "neutral" in text_lower:
            return 2
        
        # Check for abbreviated forms
        if text_lower.startswith("ent"):
            return 0
        if text_lower.startswith("cont") or text_lower.startswith("contr"):
            return 1
        if text_lower.startswith("neu"):
            return 2
        
        # Default to neutral if unclear
        return 2
    
    def _parse_sarcasm_label(self, text: str) -> int:
        """
        Parse model output to extract sarcasm label.
        Returns 1 for Sarcastic, 0 for Not Sarcastic.
        """
        text_lower = text.lower().strip()
        
        # Check for explicit labels
        if "not sarcastic" in text_lower or "not-sarcastic" in text_lower:
            return 0
        if "sarcastic" in text_lower:
            return 1
        
        # Check for abbreviated forms
        if text_lower.startswith("not"):
            return 0
        if text_lower.startswith("sarc"):
            return 1
        
        # Default to not sarcastic if unclear
        return 0
    
    def _parse_topic_label(self, text: str) -> int:
        """
        Parse model output to extract topic classification label.
        Returns the index of the matched topic in self.label_list.
        If no match, returns 0 (first topic as default).
        """
        if not self.label_list:
            return 0
        
        text_lower = text.lower().strip()
        
        # Try exact match first
        for i, label in enumerate(self.label_list):
            if label.lower() == text_lower:
                return i
        
        # Try partial match (if output contains the label)
        for i, label in enumerate(self.label_list):
            if label.lower() in text_lower:
                return i
        
        # Try if label contains the output (for abbreviated forms)
        for i, label in enumerate(self.label_list):
            if text_lower in label.lower():
                return i
        
        # Default to first topic if no match
        return 0
    
    def _parse_output(self, text: str) -> int:
        """
        Route to appropriate parsing function based on task type.
        """
        if self.task_type == "sst2":
            return self._parse_sentiment(text)
        elif self.task_type == "nli":
            return self._parse_nli_label(text)
        elif self.task_type == "sarcasm":
            return self._parse_sarcasm_label(text)
        elif self.task_type == "topic":
            return self._parse_topic_label(text)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def query_gpt2(self, full_input: str) -> int:
        """Query GPT-2 model.
        GPT-2 uses simple concatenated format (no special system/user roles)
        """
        inputs = self.tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return self._parse_output(response)
    
    def query_gpt35(self, full_input: str) -> int:
        """Query GPT-3.5 via OpenAI API.
        Uses Iris's format: system role has instructions+tokens, user role has just the sentence.
        Expected format: "You are a sentimental analysis. {prompt} Give me Positive or Negative\n{text}"
        """
        try:
            # Split on the newline - first part is system, second part is user content
            parts = full_input.split('\n', 1)
            if len(parts) == 2:
                system_instruction = parts[0].strip()  # "You are... {tokens}... Give me Positive or Negative"
                user_content = parts[1].strip()         # Just the sentence
            else:
                # Fallback: treat everything as user input with default system
                system_instruction = "You are a sentimental analysis. Give me Positive or Negative"
                user_content = full_input
            
            response = self.openai.ChatCompletion.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=10,
                temperature=0
            )
            
            text = response.choices[0].message.content
            return self._parse_output(text)
        except Exception as e:
            print(f"Warning: GPT-3.5 API error: {e}, defaulting to 0")
            return 0
    
    def query_llama(self, full_input: str) -> int:
        """Query LLaMA model.
        Uses Iris's format with LLaMA-2-chat special tokens.
        Expected format: "You are a sentimental analysis. {prompt} Give me Positive or Negative\n{text}"
        LLaMA-2-chat format: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
        """
        # Split on the newline - first part is system, second part is user content
        parts = full_input.split('\n', 1)
        if len(parts) == 2:
            system_instruction = parts[0].strip()  # "You are... {tokens}... Give me Positive or Negative"
            user_content = parts[1].strip()         # Just the sentence
            
            # Format for LLaMA-2-chat with proper special tokens
            formatted_prompt = f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{user_content} [/INST]"
        else:
            # Fallback: use simple format without special tokens
            formatted_prompt = full_input
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return self._parse_output(response)
    
    def query_one(self, full_input: str) -> int:
        """
        Query the model with caching.
        
        Args:
            full_input: Complete input string including prompt and text
            
        Returns:
            Binary label (0 or 1)
        """
        k = self._key(full_input)
        if k in self.cache:
            return self.cache[k]
        
        # Query based on model type
        if self.model_type == "gpt2":
            label = self.query_gpt2(full_input)
        elif self.model_type == "gpt3.5":
            label = self.query_gpt35(full_input)
        elif self.model_type == "llama":
            label = self.query_llama(full_input)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.cache[k] = label
        return label
    
    def predict_batch(self, inputs: List[str]) -> List[int]:
        """Predict labels for a batch of inputs."""
        return [self.query_one(x) for x in inputs]
    
    def build_inputs(self, prompt_text: str, texts: List[str], 
                     premises: Optional[List[str]] = None) -> List[str]:
        """
        Build full input strings from prompt and texts using Iris's format.
        Supports different task types with appropriate formatting.
        
        Args:
            prompt_text: Trigger tokens/prompt
            texts: List of sentences/hypotheses to classify
            premises: List of premise sentences (required for NLI task)
            
        Returns:
            List of formatted input strings
            
        Examples:
            SST-2: "You are a sentimental analysis. {prompt} Give me Positive or Negative\n{text}"
            NLI: "You are a NLI system. {prompt} Give me...\nPremise: {premise}\nHypothesis: {text}"
            Sarcasm: "You are a sarcasm detection. {prompt} Give me...\n{text}"
            Topic: "You are a topic classification. {prompt} Choose from: ...\n{text}"
        """
        result = []
        
        for i, text in enumerate(texts):
            # For NLI task, format premise and hypothesis
            if self.task_type == "nli":
                if not premises or i >= len(premises):
                    raise ValueError("NLI task requires premises for each text")
                formatted_text = f"Premise: {premises[i]}\nHypothesis: {text}"
            else:
                formatted_text = text
            
            # Use simple_template which already has task-specific instruction
            result.append(self.simple_template.format(prompt=prompt_text, text=formatted_text))
        
        return result
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "model_type": self.model_type
        }


# For backward compatibility with existing code
class BlackBoxEvaluator(BlackBoxModelEvaluator):
    """Alias for backward compatibility."""
    
    def __init__(self, model_type: str = "gpt2", model_id: Optional[str] = None,
                 task_type: str = "sst2", label_list: Optional[List[str]] = None):
        super().__init__(model_type, model_id, task_type, label_list)


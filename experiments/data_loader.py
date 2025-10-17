"""
Universal Data Loader for Multi-Task Experiments
Supports: SST-2, NLI, Sarcasm, Topic Classification
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, List
import os


def load_vocab(vocab_path: str) -> List[str]:
    """Load vocabulary from file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    return vocab


def parse_nli_sentence(sentence: str, separator: str = " B: ") -> Tuple[str, str]:
    """
    Parse NLI sentence into premise and hypothesis.
    Expected format: "A: premise B: hypothesis"
    """
    if separator not in sentence:
        # Fallback: try to find "A:" and "B:"
        if "A:" in sentence and "B:" in sentence:
            parts = sentence.split("B:")
            premise = parts[0].replace("A:", "").strip()
            hypothesis = parts[1].strip()
            return premise, hypothesis
        else:
            # Cannot parse, return as is
            return sentence, sentence
    
    parts = sentence.split(separator, 1)
    premise = parts[0].replace("A:", "").strip()
    hypothesis = parts[1].strip() if len(parts) > 1 else ""
    return premise, hypothesis


def load_task_dataset(task_config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load dataset for a specific task.
    
    Args:
        task_config: Task configuration dictionary from multi_task_config.py
        
    Returns:
        Tuple of (train_df, val_df, test_df, vocab)
    """
    dataset_path = task_config["dataset_path"]
    vocab_path = task_config["vocab_path"]
    
    # Load vocabulary
    # IRIS FLAG: 
    print(f"Loading vocabulary from: {vocab_path}")
    vocab = load_vocab(vocab_path)
    
    # Load CSV files
    train_path = os.path.join(dataset_path, task_config["train_file"])
    val_path = os.path.join(dataset_path, task_config["val_file"])
    test_path = os.path.join(dataset_path, task_config["test_file"])
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Process NLI data - split premise and hypothesis
    if task_config["task_type"] == "nli":
        separator = task_config.get("premise_hypothesis_separator", " B: ")
        
        for df in [train_df, val_df, test_df]:
            if "sentence" in df.columns:
                # Parse into premise and hypothesis
                parsed = df["sentence"].apply(lambda x: parse_nli_sentence(x, separator))
                df["premise"] = parsed.apply(lambda x: x[0])
                df["hypothesis"] = parsed.apply(lambda x: x[1])
    
    print(f"Loaded {task_config['name']} dataset:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Vocab: {len(vocab)} words")
    
    return train_df, val_df, test_df, vocab


def get_texts_and_labels(df: pd.DataFrame, task_config: dict) -> Tuple[List[str], List[int], List[str]]:
    """
    Extract texts and labels from dataframe based on task type.
    
    Args:
        df: DataFrame with data
        task_config: Task configuration
        
    Returns:
        Tuple of (texts, labels, premises) 
        - For NLI: texts=hypotheses, premises=premises
        - For others: texts=sentences, premises=None
    """
    label_col = task_config["label_column"]
    labels = df[label_col].tolist()
    
    if task_config["task_type"] == "nli":
        # NLI: return hypotheses as texts, premises separately
        texts = df["hypothesis"].tolist()
        premises = df["premise"].tolist()
    else:
        # Other tasks: single sentence
        text_col = task_config["text_columns"][0]
        texts = df[text_col].tolist()
        premises = None
    
    return texts, labels, premises


# Quick test function
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from multi_task_config import TASKS
    
    print("="*80)
    print("Testing Data Loader for All Tasks")
    print("="*80)
    
    for task_name, task_config in TASKS.items():
        print(f"\n{task_name.upper()}:")
        print("-"*80)
        try:
            train_df, val_df, test_df, vocab = load_task_dataset(task_config)
            
            # Test get_texts_and_labels
            texts, labels, premises = get_texts_and_labels(val_df.head(3), task_config)
            print(f"\nSample data:")
            for i in range(min(3, len(texts))):
                if premises:
                    print(f"  [{i+1}] Label={labels[i]}")
                    print(f"      Premise: {texts[i][:60]}...")
                    print(f"      Hypothesis: {premises[i][:60]}...")
                else:
                    print(f"  [{i+1}] Label={labels[i]}: {texts[i][:80]}...")
            
            print(f"[OK] {task_name}")
        except Exception as e:
            print(f"[ERROR] {task_name}: {e}")
    
    print("\n" + "="*80)


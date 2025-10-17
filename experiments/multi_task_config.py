"""
Task-Specific Configuration
"""

from common_config import MODELS, ALGORITHMS, TRIGGER_LENGTHS, EXPERIMENT_CONFIG

# Task-specific configurations
TASKS = {
    "sst2": {
        "name": "SST-2 Sentiment Analysis",
        "task_type": "sst2",
        "dataset_path": "../Dataset/SST-2",
        "vocab_path": "../Dataset/SST-2/vcand.txt",
        "num_labels": 2,
        "label_names": ["Negative", "Positive"],
        "label_column": "label",
        "text_columns": ["sentence"],  # Single sentence
        "train_file": "train.csv",
        "val_file": "val.csv",
        "test_file": "test.csv"
    },
    "nli": {
        "name": "Natural Language Inference",
        "task_type": "nli",
        "dataset_path": "../Dataset/NLI",
        "vocab_path": "../Dataset/NLI/vcand.txt",
        "num_labels": 3,
        "label_names": ["Entailment", "Contradiction", "Neutral"],
        "label_column": "label",
        "text_columns": ["sentence"],  # Will be split into premise/hypothesis
        "premise_hypothesis_separator": " B: ",  # "A: premise B: hypothesis"
        "train_file": "train.csv",
        "val_file": "val.csv",
        "test_file": "test.csv"
    },
    "sarcasm": {
        "name": "Sarcasm Detection",
        "task_type": "sarcasm",
        "dataset_path": "./Dataset/Sarcasm",
        "vocab_path": "./Dataset/Sarcasm/vcand.txt",
        "num_labels": 2,
        "label_names": ["Not Sarcastic", "Sarcastic"],
        "label_column": "label",
        "text_columns": ["sentence"],  # Single sentence
        "train_file": "train.csv",
        "val_file": "validation.csv",
        "test_file": "test.csv"
    },
    "topic": {
        "name": "Topic Classification",
        "task_type": "topic",
        "dataset_path": "../Dataset/Topic Classification",
        "vocab_path": "../Dataset/Topic Classification/vcand.txt",
        "num_labels": 4,
        "label_names": ["World", "Sports", "Business", "Tech"],  # AG News topics
        "label_column": "label",
        "text_columns": ["sentence"],  # Single sentence
        "train_file": "train.csv",
        "val_file": "val.csv",
        "test_file": "test.csv"
    }
}

# Default task (for backward compatibility)
DEFAULT_TASK = "sst2"

# Note: MODELS, ALGORITHMS, TRIGGER_LENGTHS, and EXPERIMENT_CONFIG 
# are imported from common_config.py to avoid duplication


def get_task_config(task_name: str):
    """Get configuration for a specific task."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASKS.keys())}")
    return TASKS[task_name]


def get_label_list(task_name: str):
    """Get label names for a specific task."""
    config = get_task_config(task_name)
    return config["label_names"]


def get_task_type(task_name: str):
    """Get task type for model evaluator."""
    config = get_task_config(task_name)
    return config["task_type"]


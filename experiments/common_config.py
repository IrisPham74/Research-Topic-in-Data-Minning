"""
Common Configuration for All Tasks
"""

# Model configurations (shared by all tasks)
MODELS = {
    "gpt2": {
        "name": "GPT-2",
        "model_id": "gpt2",
        "type": "huggingface"
    },
    "gpt3.5": {
        "name": "GPT-3.5",
        "model_id": "gpt-3.5-turbo",
        "type": "openai",
        "api_key_env": "OPENAI_API_KEY"
    },
    "llama": {
        "name": "LLaMA",
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "type": "huggingface"
    }
}

# Algorithm configurations (shared by all tasks)
ALGORITHMS = {
    "greedy": {
        "name": "Multi-Start Greedy",
        "search_algo": "multistart",
        "params": {
            "restarts": 5,
            "gens": 20,
            "pop": 40,
            "early_patience": 5
        }
    },
    "evolutionary": {
        "name": "Evolutionary",
        "search_algo": "evo",
        "params": {
            "pop": 40,
            "elite": 4,
            "gens": 20,
            "mut_p": 0.5,
            "cx_p": 0.8,
            "early_patience": 5
        }
    },
    "bandit": {
        "name": "Bandit",
        "search_algo": "bandit",
        "params": {
            "iters": 200,
            "epsilon": 0.2,
            "pop": 40
        }
    }
}

# Trigger length configurations (shared by all tasks)
TRIGGER_LENGTHS = {
    "trigger1": 3,
    "trigger2": 5,
    "trigger3": 7,
    "trigger4": 9
}

# Experiment settings (shared by all tasks)
EXPERIMENT_CONFIG = {
    "seed": 42,
    "max_tokens": 16,
    "output_dir": "experiments/results",
    "checkpoint_dir": "experiments/checkpoints"
}


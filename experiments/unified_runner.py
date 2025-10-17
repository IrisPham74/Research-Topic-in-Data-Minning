"""
Unified Experiment Runner
A unified runner for experiments - each person runs their assigned task

Design principles:
1. Each person is responsible for one task (divided by task)
2. Can choose different models (GPT-2, GPT-3.5, LLaMA)
3. Can choose different algorithms (Evolutionary, Greedy, Bandit)
4. Unified experiment management and reporting functionality

Usage:
  python unified_runner.py --task sst2 --model gpt2 --algorithm evolutionary
  
Note: Each person only runs their assigned task, no need for cross-task experiments
"""

import argparse
import json
import sys
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GAConfig, BanditConfig
from common_config import MODELS, ALGORITHMS, TRIGGER_LENGTHS, EXPERIMENT_CONFIG
from multi_task_config import TASKS, get_label_list, get_task_type
from data_loader import load_task_dataset
from search_algorithms import (
    ModelAwareEvolutionarySearch,
    ModelAwareMultiStartGreedySearch,
    ModelAwareBanditSearch
)


class ExperimentRunner:
    """Unified experiment runner"""
    
    def __init__(self, task, model, algorithm, num_trigger=3, seed=42):
        self.task = task
        self.model = model
        self.algorithm = algorithm
        self.num_trigger = num_trigger
        self.seed = seed
        
        # Get configurations
        self.task_config = TASKS[task]
        self.model_config = MODELS[model]
        self.algo_config = ALGORITHMS[algorithm]
        
        # Get task-specific info
        self.task_type = get_task_type(task)
        self.label_list = get_label_list(task) if task == "topic" else None
        
        self.start_time = None
        self.end_time = None
    
    def print_header(self):
        """Print experiment information header"""
        print("="*80)
        print("[START] Unified Experiment Runner")
        print("="*80)
        print(f"[TASK] Task:      {self.task_config['name']} ({self.task})")
        print(f"[MODEL] Model:     {self.model_config['name']}")
        print(f"[ALGO] Algorithm: {self.algo_config['name']}")
        print(f"[PARAM] Triggers:  {self.num_trigger} tokens")
        print(f"[PARAM] Seed:      {self.seed}")
        print("="*80)
    
    def load_data(self):
        """Load dataset"""
        print("\n[LOAD] Loading dataset...")
        self.train_df, self.val_df, self.test_df, self.vocab = load_task_dataset(self.task_config)
        print(f"[OK] Dataset loaded successfully")
    
    def create_search_algorithm(self):
        """Create search algorithm"""
        print(f"\n[INIT] Initializing {self.algo_config['name']}...")
        
        if self.algo_config["search_algo"] == "evo":
            cfg = GAConfig(
                pop=self.algo_config["params"]["pop"],
                elite=self.algo_config["params"]["elite"],
                gens=self.algo_config["params"]["gens"],
                max_tokens=16,
                mut_p=self.algo_config["params"]["mut_p"],
                cx_p=self.algo_config["params"]["cx_p"],
                seed=self.seed,
                early_patience=self.algo_config["params"]["early_patience"],
                num_trigger=self.num_trigger,
                search_algo="evo"
            )
            search = ModelAwareEvolutionarySearch(
                self.vocab, self.val_df, self.test_df, cfg,
                model_type=self.model,
                model_id=self.model_config["model_id"],
                task_type=self.task_type,
                label_list=self.label_list
            )
        
        elif self.algo_config["search_algo"] == "multistart":
            cfg = GAConfig(
                pop=self.algo_config["params"]["pop"],
                elite=4,
                gens=self.algo_config["params"]["gens"],
                max_tokens=16,
                seed=self.seed,
                early_patience=self.algo_config["params"]["early_patience"],
                num_trigger=self.num_trigger,
                search_algo="multi"
            )
            search = ModelAwareMultiStartGreedySearch(
                self.vocab, self.val_df, self.test_df, cfg,
                model_type=self.model,
                model_id=self.model_config["model_id"],
                restarts=self.algo_config["params"]["restarts"],
                task_type=self.task_type,
                label_list=self.label_list
            )
        
        elif self.algo_config["search_algo"] == "bandit":
            cfg = BanditConfig(
                pop=self.algo_config["params"]["pop"],
                iters=self.algo_config["params"]["iters"],
                epsilon=self.algo_config["params"]["epsilon"],
                seed=self.seed,
                num_trigger=self.num_trigger,
                search_algo="bandit"
            )
            search = ModelAwareBanditSearch(
                self.vocab, self.val_df, self.test_df, cfg,
                model_type=self.model,
                model_id=self.model_config["model_id"],
                task_type=self.task_type,
                label_list=self.label_list
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        print(f"[OK] {self.algo_config['name']} initialized")
        return search
    
    def run_experiment(self):
        self.print_header()
        self.load_data()
        search = self.create_search_algorithm()
        
        print(f"\n[RUN] Starting optimization...")
        print("[INFO] This may take a while...\n")
        
        self.start_time = time.time()
        
        try:
            result = search.run()
            self.end_time = time.time()
            
            return self.process_results(result)
            
        except Exception as e:
            print(f"\n[ERROR] Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_results(self, result):
        """Process and display results"""
        elapsed_time = self.end_time - self.start_time
        
        print("\n" + "="*80)
        print("[DONE] EXPERIMENT COMPLETED")
        print("="*80)
        print(f"[TASK] Task:         {self.task_config['name']}")
        print(f"[MODEL] Model:        {self.model_config['name']}")
        print(f"[ALGO] Algorithm:    {self.algo_config['name']}")
        print(f"[RESULT] Best Prompt:  {result['best_prompt']}")
        print(f"[RESULT] Val Accuracy: {result['val_acc']:.4f}")
        print(f"[RESULT] Test Accuracy: {result['test_acc']:.4f}")
        print(f"[TIME] Time Taken:   {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
        print("="*80)
        
        # Prepare result data
        result_data = {
            "task": self.task,
            "task_name": self.task_config['name'],
            "model": self.model,
            "model_name": self.model_config['name'],
            "algorithm": self.algorithm,
            "algorithm_name": self.algo_config['name'],
            "num_trigger": self.num_trigger,
            "seed": self.seed,
            "best_prompt": result['best_prompt'],
            "val_accuracy": result['val_acc'],
            "test_accuracy": result['test_acc'],
            "time_seconds": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result_data
    
    def save_results(self, result_data, output_file=None):
        """Save results"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/{self.task}_{self.model}_{self.algorithm}_{timestamp}.json"
        
        # Ensure results directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n[SAVED] Results saved to: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner - Each person runs their assigned task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SST-2 task with GPT-2 and evolutionary algorithm
  python unified_runner.py --task sst2 --model gpt2 --algorithm evolutionary
  
  # Run NLI task with GPT-3.5 and greedy algorithm
  python unified_runner.py --task nli --model gpt3.5 --algorithm greedy
  
  # Run sarcasm detection with LLaMA and bandit algorithm
  python unified_runner.py --task sarcasm --model llama --algorithm bandit
  
  # Run topic classification with GPT-2 and evolutionary algorithm
  python unified_runner.py --task topic --model gpt2 --algorithm evolutionary

        """
    )
    
    # Required arguments
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASKS.keys()),
                        help="Your assigned task")
    
    # Optional arguments
    parser.add_argument("--model", type=str, default="gpt2",
                        choices=list(MODELS.keys()),
                        help="Model to use (default: gpt2)")
    
    parser.add_argument("--algorithm", type=str, default="evolutionary",
                        choices=list(ALGORITHMS.keys()),
                        help="Search algorithm (default: evolutionary)")
    
    parser.add_argument("--num_trigger", type=int, default=3,
                        help="Number of trigger tokens (default: 3)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Create and run experiment
    runner = ExperimentRunner(
        task=args.task,
        model=args.model,
        algorithm=args.algorithm,
        num_trigger=args.num_trigger,
        seed=args.seed
    )
    
    result_data = runner.run_experiment()
    
    if result_data:
        runner.save_results(result_data, args.output)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())


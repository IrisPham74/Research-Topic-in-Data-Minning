"""
Universal Timing Analysis
A universal timing analysis tool that supports all tasks.

Reads all result JSON files from a specified directory and automatically
groups and analyzes timing data by task.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_config import MODELS, ALGORITHMS, TRIGGER_LENGTHS


class UniversalTimingAnalyzer:
    """Universal timing analyzer for experiment results"""
    
    def __init__(self, results_dir="experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_by_task = defaultdict(list)
        self.task_names = {}
    
    def load_all_results(self):
        """Load all result JSON files from the specified directory"""
        if not self.results_dir.exists():
            print(f"[ERROR] Results directory not found: {self.results_dir}")
            print("   Run some experiments first to generate results.")
            return False
        
        json_files = list(self.results_dir.glob("*.json"))
        
        if not json_files:
            print(f"[ERROR] No result files found in: {self.results_dir}")
            print("   Run some experiments first.")
            return False
        
        print(f"\n[TIMING] Loading timing data from: {self.results_dir}")
        print(f"   Found {len(json_files)} result files\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract task information
                    task = data.get('task', 'unknown')
                    task_name = data.get('task_name', task)
                    
                    # Only include if has timing info
                    if 'time_seconds' in data:
                        self.results_by_task[task].append(data)
                        self.task_names[task] = task_name
                    
            except Exception as e:
                print(f"[WARNING] Failed to load {json_file.name}: {e}")
        
        total_with_timing = sum(len(r) for r in self.results_by_task.values())
        print(f"[OK] Loaded {total_with_timing} results with timing data")
        print(f"[OK] Found {len(self.results_by_task)} tasks: {', '.join(self.results_by_task.keys())}\n")
        
        return True
    
    def format_time(self, seconds):
        """Format time for display"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = seconds / 60
            secs = seconds % 60
            return f"{int(mins)}m {int(secs)}s"
        else:
            hours = seconds / 3600
            mins = (seconds % 3600) / 60
            return f"{int(hours)}h {int(mins)}m"
    
    def analyze_by_model(self, task):
        """Analyze timing by model"""
        results = self.results_by_task[task]
        task_name = self.task_names.get(task, task)
        
        if not results:
            return
        
        print(f"\n[MODEL] Timing Analysis by Model - {task_name}")
        print("="*80)
        
        # Group by model
        by_model = defaultdict(list)
        for result in results:
            model = result.get('model', 'unknown')
            time_sec = result.get('time_seconds', 0)
            by_model[model].append(time_sec)
        
        # Print table
        print(f"{'Model':<20} {'Count':<10} {'Mean':<15} {'Min':<15} {'Max':<15} {'Total':<15}")
        print("-"*80)
        
        for model in sorted(by_model.keys()):
            times = by_model[model]
            model_name = MODELS.get(model, {}).get('name', model)
            
            mean_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)
            
            print(f"{model_name:<20} {len(times):<10} {self.format_time(mean_time):<15} "
                  f"{self.format_time(min_time):<15} {self.format_time(max_time):<15} "
                  f"{self.format_time(total_time):<15}")
        
        print("-"*80)
    
    def analyze_by_algorithm(self, task):
        """Analyze timing by algorithm"""
        results = self.results_by_task[task]
        task_name = self.task_names.get(task, task)
        
        if not results:
            return
        
        print(f"\n[ALGORITHM] Timing Analysis by Algorithm - {task_name}")
        print("="*80)
        
        # Group by algorithm
        by_algo = defaultdict(list)
        for result in results:
            algo = result.get('algorithm', 'unknown')
            time_sec = result.get('time_seconds', 0)
            by_algo[algo].append(time_sec)
        
        # Print table
        print(f"{'Algorithm':<20} {'Count':<10} {'Mean':<15} {'Min':<15} {'Max':<15} {'Total':<15}")
        print("-"*80)
        
        for algo in sorted(by_algo.keys()):
            times = by_algo[algo]
            algo_name = ALGORITHMS.get(algo, {}).get('name', algo)
            
            mean_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)
            
            print(f"{algo_name:<20} {len(times):<10} {self.format_time(mean_time):<15} "
                  f"{self.format_time(min_time):<15} {self.format_time(max_time):<15} "
                  f"{self.format_time(total_time):<15}")
        
        print("-"*80)
    
    def analyze_by_trigger_length(self, task):
        """Analyze timing by trigger token length"""
        results = self.results_by_task[task]
        task_name = self.task_names.get(task, task)
        
        if not results:
            return
        
        print(f"\n[TRIGGER] Timing Analysis by Trigger Length - {task_name}")
        print("="*80)
        
        # Group by trigger length
        by_trigger = defaultdict(list)
        for result in results:
            num_trigger = result.get('num_trigger', 0)
            time_sec = result.get('time_seconds', 0)
            by_trigger[num_trigger].append(time_sec)
        
        # Print table
        print(f"{'Trigger Length':<20} {'Count':<10} {'Mean':<15} {'Min':<15} {'Max':<15} {'Total':<15}")
        print("-"*80)
        
        for trigger_len in sorted(by_trigger.keys()):
            times = by_trigger[trigger_len]
            
            mean_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)
            
            print(f"{trigger_len} tokens{'':<11} {len(times):<10} {self.format_time(mean_time):<15} "
                  f"{self.format_time(min_time):<15} {self.format_time(max_time):<15} "
                  f"{self.format_time(total_time):<15}")
        
        print("-"*80)
    
    def analyze_algorithm_comparison(self, task):
        """Algorithm efficiency comparison (time vs accuracy)"""
        results = self.results_by_task[task]
        task_name = self.task_names.get(task, task)
        
        if not results:
            return
        
        print(f"\n[EFFICIENCY] Algorithm Efficiency (Time vs Accuracy) - {task_name}")
        print("="*80)
        
        # Group by algorithm
        by_algo = defaultdict(lambda: {'times': [], 'accs': []})
        for result in results:
            algo = result.get('algorithm', 'unknown')
            time_sec = result.get('time_seconds', 0)
            test_acc = result.get('test_accuracy', 0.0)
            
            by_algo[algo]['times'].append(time_sec)
            by_algo[algo]['accs'].append(test_acc)
        
        # Print table
        print(f"{'Algorithm':<20} {'Mean Time':<15} {'Mean Accuracy':<15} {'Efficiency*':<15}")
        print("-"*80)
        
        for algo in sorted(by_algo.keys()):
            algo_name = ALGORITHMS.get(algo, {}).get('name', algo)
            times = by_algo[algo]['times']
            accs = by_algo[algo]['accs']
            
            mean_time = sum(times) / len(times)
            mean_acc = sum(accs) / len(accs)
            
            # Efficiency: accuracy per minute
            efficiency = (mean_acc * 100) / (mean_time / 60)
            
            print(f"{algo_name:<20} {self.format_time(mean_time):<15} "
                  f"{mean_acc:.4f}{'':<9} {efficiency:.2f} pts/min{'':<2}")
        
        print("-"*80)
        print("* Efficiency = (Accuracy * 100) / (Time in minutes)")
    
    def analyze_model_comparison(self, task):
        """Model efficiency comparison (time vs accuracy)"""
        results = self.results_by_task[task]
        task_name = self.task_names.get(task, task)
        
        if not results:
            return
        
        print(f"\n[EFFICIENCY] Model Efficiency (Time vs Accuracy) - {task_name}")
        print("="*80)
        
        # Group by model
        by_model = defaultdict(lambda: {'times': [], 'accs': []})
        for result in results:
            model = result.get('model', 'unknown')
            time_sec = result.get('time_seconds', 0)
            test_acc = result.get('test_accuracy', 0.0)
            
            by_model[model]['times'].append(time_sec)
            by_model[model]['accs'].append(test_acc)
        
        # Print table
        print(f"{'Model':<20} {'Mean Time':<15} {'Mean Accuracy':<15} {'Efficiency*':<15}")
        print("-"*80)
        
        for model in sorted(by_model.keys()):
            model_name = MODELS.get(model, {}).get('name', model)
            times = by_model[model]['times']
            accs = by_model[model]['accs']
            
            mean_time = sum(times) / len(times)
            mean_acc = sum(accs) / len(accs)
            
            # Efficiency: accuracy per minute
            efficiency = (mean_acc * 100) / (mean_time / 60)
            
            print(f"{model_name:<20} {self.format_time(mean_time):<15} "
                  f"{mean_acc:.4f}{'':<9} {efficiency:.2f} pts/min{'':<2}")
        
        print("-"*80)
        print("* Efficiency = (Accuracy * 100) / (Time in minutes)")
    
    def generate_timing_report(self, task=None):
        """Generate complete timing analysis report"""
        if not self.load_all_results():
            return
        
        tasks_to_report = [task] if task else sorted(self.results_by_task.keys())
        
        print("\n" + "="*80)
        print("[TIMING] UNIVERSAL TIMING ANALYSIS REPORT")
        print("="*80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results directory: {self.results_dir}")
        print("="*80)
        
        for task in tasks_to_report:
            if task not in self.results_by_task:
                print(f"\n[WARNING] No timing data found for task: {task}")
                continue
            
            self.analyze_by_model(task)
            self.analyze_by_algorithm(task)
            self.analyze_by_trigger_length(task)
            self.analyze_algorithm_comparison(task)
            self.analyze_model_comparison(task)
            print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Timing Analyzer - Analyze timing for all tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze timing for all tasks
  python universal_timing.py
  
  # Analyze timing for a specific task only
  python universal_timing.py --task sst2
  
  # Specify results directory
  python universal_timing.py --results-dir experiments/results
        """
    )
    
    parser.add_argument("--task", type=str, default=None,
                        help="Analyze only a specific task (default: all tasks)")
    
    parser.add_argument("--results-dir", type=str, default="experiments/results",
                        help="Results directory (default: experiments/results)")
    
    args = parser.parse_args()
    
    # Create timing analyzer
    analyzer = UniversalTimingAnalyzer(results_dir=args.results_dir)
    
    # Generate timing report
    analyzer.generate_timing_report(task=args.task)


if __name__ == "__main__":
    main()


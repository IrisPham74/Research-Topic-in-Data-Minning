"""
Universal Report Generator
通用报告生成工具 - 支持所有任务

可以读取指定目录下的所有结果JSON文件，自动按task分组生成报告
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_config import MODELS, ALGORITHMS, TRIGGER_LENGTHS


class UniversalReportGenerator:
    """通用报告生成器"""
    
    def __init__(self, results_dir="experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_by_task = defaultdict(list)
        self.task_names = {}
    
    def load_all_results(self):
        """加载指定目录下的所有结果JSON文件"""
        if not self.results_dir.exists():
            print(f"[ERROR] Results directory not found: {self.results_dir}")
            print("   Run some experiments first to generate results.")
            return False
        
        json_files = list(self.results_dir.glob("*.json"))
        
        if not json_files:
            print(f"[ERROR] No result files found in: {self.results_dir}")
            print("   Run some experiments first.")
            return False
        
        print(f"\n[LOADING] Loading results from: {self.results_dir}")
        print(f"   Found {len(json_files)} result files\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract task information
                    task = data.get('task', 'unknown')
                    task_name = data.get('task_name', task)
                    
                    self.results_by_task[task].append(data)
                    self.task_names[task] = task_name
                    
            except Exception as e:
                print(f"[WARNING] Failed to load {json_file.name}: {e}")
        
        print(f"[OK] Loaded {sum(len(r) for r in self.results_by_task.values())} results")
        print(f"[OK] Found {len(self.results_by_task)} tasks: {', '.join(self.results_by_task.keys())}\n")
        
        return True
    
    def print_summary_table(self, task):
        """打印某个任务的汇总表格"""
        results = self.results_by_task[task]
        task_name = self.task_names.get(task, task)
        
        if not results:
            print(f"No results for task: {task}")
            return
        
        # Organize results by model, algorithm, trigger length
        organized = defaultdict(lambda: defaultdict(dict))
        
        for result in results:
            model = result.get('model', 'unknown')
            algo = result.get('algorithm', 'unknown')
            num_trigger = result.get('num_trigger', 0)
            test_acc = result.get('test_accuracy', 0.0)
            
            organized[model][algo][num_trigger] = test_acc
        
        # Print header
        print("="*80)
        print(f"[TASK] Task: {task_name} ({task})")
        print("="*80)
        
        # Print table for each model
        for model_key in sorted(organized.keys()):
            model_name = MODELS.get(model_key, {}).get('name', model_key)
            print(f"\n[MODEL] Model: {model_name}")
            print("-"*80)
            print(f"{'Trigger':<15} {'Greedy':<15} {'Evolutionary':<15} {'Bandit':<15}")
            print("-"*80)
            
            # Sort trigger lengths
            all_triggers = set()
            for algo_results in organized[model_key].values():
                all_triggers.update(algo_results.keys())
            
            for trigger_len in sorted(all_triggers):
                row = [f"{trigger_len} tokens"]
                
                for algo_key in ['greedy', 'evolutionary', 'bandit']:
                    acc = organized[model_key].get(algo_key, {}).get(trigger_len, None)
                    if acc is not None:
                        row.append(f"{acc:.4f}")
                    else:
                        row.append("-")
                
                print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
            
            print("-"*80)
    
    def print_best_results(self, task):
        """打印某个任务的最佳结果"""
        results = self.results_by_task[task]
        task_name = self.task_names.get(task, task)
        
        if not results:
            return
        
        # Find best result
        best_result = max(results, key=lambda x: x.get('test_accuracy', 0.0))
        
        print(f"\n[BEST] Best Result for {task_name}:")
        print("-"*80)
        print(f"  Model:        {best_result.get('model_name', 'unknown')}")
        print(f"  Algorithm:    {best_result.get('algorithm_name', 'unknown')}")
        print(f"  Triggers:     {best_result.get('num_trigger', 0)} tokens")
        print(f"  Best Prompt:  {best_result.get('best_prompt', 'N/A')}")
        print(f"  Val Accuracy: {best_result.get('val_accuracy', 0.0):.4f}")
        print(f"  Test Accuracy: {best_result.get('test_accuracy', 0.0):.4f}")
        if 'time_seconds' in best_result:
            time_min = best_result['time_seconds'] / 60
            print(f"  Time Taken:   {time_min:.2f} minutes")
    
    def print_statistics(self, task):
        """打印某个任务的统计信息"""
        results = self.results_by_task[task]
        task_name = self.task_names.get(task, task)
        
        if not results:
            return
        
        print(f"\n[STATS] Statistics for {task_name}:")
        print("-"*80)
        
        # Count by model
        model_counts = defaultdict(int)
        for r in results:
            model_counts[r.get('model', 'unknown')] += 1
        
        print("  Experiments by Model:")
        for model, count in sorted(model_counts.items()):
            model_name = MODELS.get(model, {}).get('name', model)
            print(f"    {model_name}: {count} experiments")
        
        # Count by algorithm
        algo_counts = defaultdict(int)
        for r in results:
            algo_counts[r.get('algorithm', 'unknown')] += 1
        
        print("\n  Experiments by Algorithm:")
        for algo, count in sorted(algo_counts.items()):
            algo_name = ALGORITHMS.get(algo, {}).get('name', algo)
            print(f"    {algo_name}: {count} experiments")
        
        # Accuracy statistics
        test_accs = [r.get('test_accuracy', 0.0) for r in results]
        if test_accs:
            print(f"\n  Test Accuracy:")
            print(f"    Mean:   {sum(test_accs)/len(test_accs):.4f}")
            print(f"    Min:    {min(test_accs):.4f}")
            print(f"    Max:    {max(test_accs):.4f}")
        
        # Time statistics
        times = [r.get('time_seconds', 0) for r in results if 'time_seconds' in r]
        if times:
            print(f"\n  Time (minutes):")
            print(f"    Mean:   {sum(times)/len(times)/60:.2f}")
            print(f"    Total:  {sum(times)/60:.2f}")
    
    def generate_report(self, task=None):
        """生成完整报告"""
        if not self.load_all_results():
            return
        
        tasks_to_report = [task] if task else sorted(self.results_by_task.keys())
        
        print("\n" + "="*80)
        print("[REPORT] UNIVERSAL EXPERIMENT REPORT")
        print("="*80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results directory: {self.results_dir}")
        print("="*80)
        
        for task in tasks_to_report:
            if task not in self.results_by_task:
                print(f"\n[WARNING] No results found for task: {task}")
                continue
            
            self.print_summary_table(task)
            self.print_best_results(task)
            self.print_statistics(task)
            print("\n")
    
    def export_to_csv(self, output_file=None):
        """导出结果为CSV格式"""
        if not self.results_by_task:
            print("No results to export")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/summary_{timestamp}.csv"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect all results
        all_results = []
        for task, results in self.results_by_task.items():
            all_results.extend(results)
        
        # Write CSV
        with open(output_path, 'w') as f:
            # Header
            f.write("task,task_name,model,model_name,algorithm,algorithm_name,")
            f.write("num_trigger,seed,best_prompt,val_accuracy,test_accuracy,")
            f.write("time_seconds,timestamp\n")
            
            # Data
            for result in all_results:
                f.write(f"{result.get('task', '')},")
                f.write(f"{result.get('task_name', '')},")
                f.write(f"{result.get('model', '')},")
                f.write(f"{result.get('model_name', '')},")
                f.write(f"{result.get('algorithm', '')},")
                f.write(f"{result.get('algorithm_name', '')},")
                f.write(f"{result.get('num_trigger', 0)},")
                f.write(f"{result.get('seed', 0)},")
                f.write(f"\"{result.get('best_prompt', '')}\",")
                f.write(f"{result.get('val_accuracy', 0.0)},")
                f.write(f"{result.get('test_accuracy', 0.0)},")
                f.write(                f"{result.get('time_seconds', 0.0)},")
                f.write(f"{result.get('timestamp', '')}\n")
        
        print(f"\n[SAVED] Results exported to: {output_path}")
        print(f"   Total entries: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Report Generator - 为所有任务生成报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成所有任务的报告
  python universal_report.py
  
  # 只生成特定任务的报告
  python universal_report.py --task sst2
  
  # 指定结果目录
  python universal_report.py --results-dir experiments/results
  
  # 导出为CSV
  python universal_report.py --export-csv
  python universal_report.py --export-csv --output results.csv
        """
    )
    
    parser.add_argument("--task", type=str, default=None,
                        help="只生成特定任务的报告 (默认: 所有任务)")
    
    parser.add_argument("--results-dir", type=str, default="experiments/results",
                        help="结果文件目录 (默认: experiments/results)")
    
    parser.add_argument("--export-csv", action="store_true",
                        help="导出为CSV格式")
    
    parser.add_argument("--output", type=str, default=None,
                        help="CSV输出文件路径")
    
    args = parser.parse_args()
    
    # Create report generator
    generator = UniversalReportGenerator(results_dir=args.results_dir)
    
    # Generate report
    generator.generate_report(task=args.task)
    
    # Export if requested
    if args.export_csv:
        generator.export_to_csv(output_file=args.output)


if __name__ == "__main__":
    main()


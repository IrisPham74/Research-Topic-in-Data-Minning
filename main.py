import argparse
import json
from config import GAConfig, BanditConfig
from utils import load_dataset_csvs, load_vocab
from evolutionary import EvolutionarySearch
from greedy import MultiStartGreedySearch
from bandit import BanditSearch


# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Prompt Optimization Runner")

    # ---- Dataset & vocab ----
    parser.add_argument("--dataset", type=str,
                        help="Path or name of dataset folder")
    parser.add_argument("--vocab_path", type=str,
                        help="Path to vocabulary file (default: Vcand/vcand.txt)")

    # ---- Common config ----
    parser.add_argument("--num_trigger", type=int, default=3,
                        help="Number of trigger tokens in prompt")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--search_algo", type=str, default="evo",
                        choices=["evo", "bandit", "multi", "multi_greedy", "multistart"],
                        help="Search algorithm")

    # ---- GA (EvolutionarySearch) parameters ----
    parser.add_argument("--pop", type=int, default=40)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--gens", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--mut_p", type=float, default=0.5)
    parser.add_argument("--ins_p", type=float, default=0.25)
    parser.add_argument("--del_p", type=float, default=0.25)
    parser.add_argument("--swap_p", type=float, default=0.25)
    parser.add_argument("--cx_p", type=float, default=0.8)
    parser.add_argument("--tourn_k", type=int, default=4)
    parser.add_argument("--early_patience", type=int, default=5)

    # ---- Bandit (BanditSearch) parameters ----
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Exploration rate for epsilon-greedy")
    parser.add_argument("--iters", type=int, default=200,
                        help="Number of search iterations")

    # ---- MultiStartGreedySearch parameters ----
    parser.add_argument("--restarts", type=int, default=5,
                        help="Number of restarts for multi-start greedy search")

    args = parser.parse_args()

    # --- Load data and vocab ---
    try:
        _, val_df, test_df = load_dataset_csvs(args.dataset)
        vocab = load_vocab(args.vocab_path)
        print(f"Loaded dataset: {args.dataset}")
        print(f"Loaded vocabulary: {len(vocab)} words")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Choose method and configure ---
    if args.search_algo == "evo":
        cfg = GAConfig(
            pop=args.pop,
            elite=args.elite,
            gens=args.gens,
            max_tokens=args.max_tokens,
            mut_p=args.mut_p,
            ins_p=args.ins_p,
            del_p=args.del_p,
            swap_p=args.swap_p,
            cx_p=args.cx_p,
            tourn_k=args.tourn_k,
            seed=args.seed,
            early_patience=args.early_patience,
            num_trigger=args.num_trigger,
            search_algo="evo"
        )
        search = EvolutionarySearch(vocab, val_df, test_df, cfg)
        print(f"Initialized Evolutionary Search with population {args.pop}, generations {args.gens}")

    elif args.search_algo == "bandit":
        cfg = BanditConfig(
            pop=args.pop,
            elite=args.elite,
            gens=args.gens,
            max_tokens=args.max_tokens,
            mut_p=args.mut_p,
            ins_p=args.ins_p,
            del_p=args.del_p,
            swap_p=args.swap_p,
            cx_p=args.cx_p,
            tourn_k=args.tourn_k,
            seed=args.seed,
            early_patience=args.early_patience,
            num_trigger=args.num_trigger,
            epsilon=args.epsilon,
            iters=args.iters,
            search_algo="bandit"
        )
        search = BanditSearch(vocab, val_df, test_df, cfg)
        print(f"Initialized Bandit Search with {args.iters} iterations, epsilon {args.epsilon}")

    elif args.search_algo in ["multi", "multi_greedy", "multistart"]:
        cfg = GAConfig(
            pop=args.pop,
            elite=args.elite,
            gens=args.gens,
            max_tokens=args.max_tokens,
            mut_p=args.mut_p,
            ins_p=args.ins_p,
            del_p=args.del_p,
            swap_p=args.swap_p,
            cx_p=args.cx_p,
            tourn_k=args.tourn_k,
            seed=args.seed,
            early_patience=args.early_patience,
            num_trigger=args.num_trigger,
            search_algo="multi"
        )
        search = MultiStartGreedySearch(vocab, val_df, test_df, cfg, restarts=args.restarts)
        print(f"Initialized Multi-Start Greedy Search with {args.restarts} restarts")

    else:
        raise ValueError(f"Unknown method: {args.search_algo}")

    # --- Run search ---
    print(f"\nStarting {args.search_algo} search...")
    try:
        result = search.run()
        print("\n" + "=" * 50)
        print("SEARCH COMPLETED")
        print("=" * 50)
        print(f"Best Prompt: {result['best_prompt']}")
        print(f"Validation Accuracy: {result['val_acc']:.4f}")
        print(f"Test Accuracy: {result['test_acc']:.4f}")
        print("=" * 50)

        output_file = f"results_{args.search_algo}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
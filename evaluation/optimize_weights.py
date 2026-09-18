import optuna
import copy
from evaluation.evaluate import load_dataset, evaluate_ranker
from task4_rank_evidence import DEFAULT_CONFIG

# 1. Load the dataset globally so it doesn't reload on every trial
dataset_path = "evaluation/sample_dataset.jsonl" 
grouped_dataset = load_dataset(dataset_path)

def objective(trial):
    """
    The objective function evaluated by Optuna.
    It suggests weights, normalizes them, and returns the nDCG score.
    """
    # Step A: Suggest float values between 0.01 and 1.0 for each weight
    w_query = trial.suggest_float("w_query", 0.01, 1.0)
    w_context = trial.suggest_float("w_context", 0.01, 1.0)
    w_evidence = trial.suggest_float("w_evidence", 0.01, 1.0)
    
    # Step B: Normalize the weights so they mathematically sum to 1.0
    total = w_query + w_context + w_evidence
    w_query /= total
    w_context /= total
    w_evidence /= total
    
    # Step C: Inject the new weights into a copy of your baseline config
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ranking_weights"] = {
        "query_relevance_weight": w_query,
        "indian_context_weight": w_context,
        "evidence_weight": w_evidence
    }
    
    # Step D: Run the evaluation pipeline
    summary = evaluate_ranker(grouped_dataset, config=config, ks=(10,))
    
    # Step E: Return the specific metric Optuna needs to maximize
    return summary["mean_nDCG@10"]

if __name__ == "__main__":
    print("Starting Weight Optimization...")
    
    # Create the study object and set the direction to "maximize" the nDCG
    study = optuna.create_study(direction="maximize")
    
    # Run 100 trials (this should take less than a minute on a small dataset)
    study.optimize(objective, n_trials=100)
    
    # Output the final learned weights
    print("\n==================================")
    print("      OPTIMAL WEIGHTS FOUND       ")
    print("==================================")
    
    best = study.best_trial
    total = sum(best.params.values())
    
    print(f"Query Relevance:  {best.params['w_query'] / total:.3f}")
    print(f"Indian Context:   {best.params['w_context'] / total:.3f}")
    print(f"Evidence Signal:  {best.params['w_evidence'] / total:.3f}")
    print("----------------------------------")
    print(f"Peak nDCG@10:     {best.value:.4f}")
    print("==================================")
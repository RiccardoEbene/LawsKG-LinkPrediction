import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.inference import inference
from scipy.stats import ks_2samp

def build_inductivity_test_set(gt_path, all_pairs_path, set_id, k=10):
    gt = pd.read_csv(gt_path)
    all_pairs = pd.read_csv(all_pairs_path)

    # compute difference between all pairs and gt pairs
    neg_pairs = all_pairs.merge(gt, on=["node_1", "node_2"], how="left", indicator=True).query('_merge == "left_only"').drop(columns=["_merge"])

    # randomly sample 1000 negative pairs
    neg_sample = neg_pairs.sample(n=len(gt)*k)

    # combine gt and negative samples
    test_set = pd.concat([gt, neg_sample], ignore_index=True)

    # shuffle the test set
    test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)
    test_set.to_csv(f"data/inductivity_test/test_set_{set_id}.csv", index=False)

import argparse
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def run_test(gt_path, all_pairs_path, k=10):
    num_trials = 5

    # Carica il ground-truth una sola volta all'inizio
    df_gt = pd.read_csv(gt_path)

    # Crea il set di frozenset per il lookup veloce delle coppie reali
    gt_pairs = set(
        frozenset(r)
        for r in df_gt[["node_1", "node_2"]].itertuples(index=False)
    )

    # Liste per salvare le statistiche di ogni iterazione
    ks_statistics = []
    p_values = []

    for i in range(num_trials):
        build_inductivity_test_set(gt_path, all_pairs_path, set_id=i, k=k)

        # Prepare paths
        input_csv = f"data/inductivity_test/test_set_{i}.csv"
        output_csv = f"output/inductivity_test/scores_{i}.csv"

        # Build args for the inference function
        args = argparse.Namespace(
            nodes_path="data/all_nodes_emb.parquet",
            edges_path="data/edges.csv",
            input_csv=input_csv,
            output_csv=output_csv,
            model_path="checkpoints/tuned_graphsage_full.pth",
        )

        try:
            # Esegue l'inference e scrive l'output_csv
            inference(args)

            # Carica i risultati dell'inference appena calcolati
            df_results = pd.read_csv(output_csv)

            df_results["is_gt"] = df_results.apply(
                lambda r: frozenset((r.node_1, r.node_2)) in gt_pairs, axis=1
            )

            # Estrai gli score
            gt_scores = df_results.loc[df_results["is_gt"], "score_prob"].values
            neg_scores = df_results.loc[
                ~df_results["is_gt"], "score_prob"
            ].values

            # Esegui il test KS a due campioni
            # alternative='less' verifica se la CDF di gt è a destra (valori più alti) di neg
            ks_res = ks_2samp(gt_scores, neg_scores, alternative="less")

            ks_statistics.append(ks_res.statistic)
            p_values.append(ks_res.pvalue)

            print(
                f"Trial {i} - KS Statistic (D): {ks_res.statistic:.4f}, p-value: {ks_res.pvalue:.4e}"
            )

        except Exception as e:
            print(f"Inference or KS test failed for trial {i}: {e}")

    if ks_statistics:
        mean_d = np.mean(ks_statistics)
        mean_p = np.mean(p_values)

        print("\n" + "=" * 40)
        print("FINAL INDUCTIVITY TEST RESULTS (AVERAGE):")
        print(f"Mean KS Statistic (D): {mean_d:.4f}")
        print(f"Mean p-value: {mean_p:.4e}")
        print("=" * 40)

        return {"mean_d": mean_d, "mean_p-value": mean_p}
    else:
        print("No successful trials to aggregate.")
        return None


def plot_score_distribution(df_path, gt_path, output_path="score_distribution.png"):
    if os.path.isdir(df_path):
        score_paths = sorted(glob.glob(os.path.join(df_path, "scores_*.csv")))
        if not score_paths:
            raise FileNotFoundError(f"No score files found in {df_path}")

        frames = [pd.read_csv(path) for path in score_paths]
        df = pd.concat(frames, ignore_index=True)
    else:
        score_paths = [df_path]
        df = pd.read_csv(df_path)

    gt = pd.read_csv(gt_path)

    gt_pairs = set(frozenset(r) for r in gt[["node_1", "node_2"]].itertuples(index=False))
    df["is_gt"] = df.apply(lambda r: frozenset((r.node_1, r.node_2)) in gt_pairs, axis=1)

    gt_scores  = df.loc[ df["is_gt"], "score_prob"]
    neg_scores = df.loc[~df["is_gt"], "score_prob"]

    plt.figure(figsize=(8, 5))
    plt.hist(neg_scores, bins=50, alpha=0.6, density=True, label=f"Non-GT (n={len(neg_scores)})", color="tomato")
    plt.hist(gt_scores,  bins=50, alpha=0.6, density=True, label=f"GT (n={len(gt_scores)})",      color="seagreen")
    plt.axvline(neg_scores.mean(), color="tomato",   linestyle="--", linewidth=1.5, label=f"Non-GT mean={neg_scores.mean():.3f}")
    plt.axvline(gt_scores.mean(),  color="seagreen", linestyle="--", linewidth=1.5, label=f"GT mean={gt_scores.mean():.3f}")
    plt.xlabel("Score Normalized")
    plt.ylabel("Density")
    plt.title(f"Score distribution across {len(score_paths)} run(s): Ground Truth vs Non-Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_score_distribution(df_path="output/inductivity_test/", gt_path="data/inductivity_test/citEUNat.csv", output_path="score_distribution.png")
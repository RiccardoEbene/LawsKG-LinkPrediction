import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_recall_curve
import numpy as np
import os
import random
import pandas as pd

def compute_loss(scores, labels, pos_weight=None):
    return F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=pos_weight)

def compute_auc(scores, labels):
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    try:
        return roc_auc_score(labels_np, scores_np)
    except ValueError:
        return 0.0

def compute_recall(scores, labels, threshold=0.5):
    probs = torch.sigmoid(scores)
    preds = (probs >= threshold).long().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return recall_score(labels_np, preds)

def compute_f1(scores, labels, threshold=0.5):
    scores_np = torch.sigmoid(scores).cpu().numpy() 
    labels_np = labels.cpu().numpy()
    
    # Calcola precision, recall e thresholds per ogni possibile taglio
    precisions, recalls, thresholds = precision_recall_curve(labels_np, scores_np)
    
    # Calcola F1 per ogni soglia
    # Nota: F1 = 2 * (P * R) / (P + R)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Trova il massimo
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx]
    
    return best_f1, best_thresh

def compute_ranking_metrics(ground_truth_csv, ranked_csv, k=50):
    ground_truth_df = pd.read_csv(ground_truth_csv)
    ranked_df = pd.read_csv(ranked_csv)

    ground_truth_pairs = list(zip(ground_truth_df["originid"].astype(str), ground_truth_df["destID"].astype(str)))
    ranked_pairs = list(zip(ranked_df["node_1"].astype(str), ranked_df["node_2"].astype(str)))
    topk_pairs = set(ranked_pairs[:k])
    ranks = {pair: idx + 1 for idx, pair in enumerate(ranked_pairs)}

    hits_at_k = sum(1 for pair in ground_truth_pairs if pair in topk_pairs)
    recall_at_k = hits_at_k / len(ground_truth_pairs)
    mrr = sum((1 / ranks[pair]) if pair in ranks else 0.0 for pair in ground_truth_pairs) / len(ground_truth_pairs)

    return {
        "hits_at_k": hits_at_k,
        "topk_percentage": recall_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
    }
    
def save_checkpoint(model, optimizer, config, node_map, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'node_map': node_map,
    }, filepath)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """Set random seeds for reproducible runs.

    This sets seeds for Python's `random`, NumPy and PyTorch, and configures
    cuDNN for deterministic behavior where possible.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN: deterministic ensures reproducible results at cost of performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Filter out edges with unknown nodes
def delete_invalid_couples(edges_df, nodes_df):

    valid_node_ids = set(nodes_df['node_id'])

    valid_edges_df = edges_df[
        edges_df['node_1'].isin(valid_node_ids) & edges_df['node_2'].isin(valid_node_ids)
    ]

    print(f"Dropped {len(edges_df) - len(valid_edges_df)} edges connecting to unknown nodes.")

    return valid_edges_df.reset_index(drop=True)


def create_added_removed_sets(old_csv, new_csv, added_out_path, removed_out_path, k=50):
    # Load and slice the top k rows
    df1 = pd.read_csv(old_csv).head(k)
    df2 = pd.read_csv(new_csv).head(k)

    # Filter rows in second not in first (Added)
    added = df2[~df2['node_id'].isin(df1['node_id'])]
    
    # Filter rows in first not in second (Removed)
    removed = df1[~df1['node_id'].isin(df2['node_id'])]

    # Save to outputs
    added.to_csv(added_out_path, index=False)
    removed.to_csv(removed_out_path, index=False)


if __name__ == "__main__":
    print(compute_ranking_metrics("data/citEUNat.csv", "output/EUNat_ranked.csv", k=10000))
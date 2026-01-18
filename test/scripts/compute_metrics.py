import pandas as pd
import numpy as np

# --- 1. Helper to get relevant rows (The "Hits") ---
def get_hits(df, target_id):
    """Returns the rows matching the Law ID (prefix) or Article ID (exact)."""
    # Force string conversion to handle IDs safely
    df['node_id'] = df['node_id'].astype(str)
    
    if '#' in str(target_id): 
        # Exact Article ID
        return df[df['node_id'] == target_id]
    else: 
        # Law ID (Prefix match)
        prefix = f"{target_id}#"
        return df[df['node_id'].str.startswith(prefix)]

# --- 2. The Metric Functions ---

def recall_at_k(df, target_id, total_gt, k):
    """Returns the percentage of the law found in the top K results."""
    hits = get_hits(df, target_id)
    found_in_k = len(hits[hits['rank'] <= k])
    return found_in_k / total_gt

def search_depth_for_recall(df, target_id, total_gt, recall_target=1.0):
    """
    Returns the Rank (depth) you must scroll to in order to find X% of the law.
    Returns -1 if that recall target is not reached in the file.
    """
    hits = get_hits(df, target_id).sort_values('rank')
    
    # How many articles constitute X%? (e.g. 0.75 * 12 = 9 articles)
    needed_count = int(np.ceil(recall_target * total_gt))
    
    if len(hits) < needed_count:
        return -1 # Target not reachable with current results
        
    # Return the rank of the Nth item (iloc is 0-indexed)
    return hits.iloc[needed_count - 1]['rank']

def first_hit_rank(df, target_id):
    """Returns the rank of the very first appearance (Best Rank)."""
    hits = get_hits(df, target_id)
    if hits.empty:
        return None
    return hits['rank'].min()

def total_recovered(df, target_id):
    """Returns simply how many relevant items are in the file."""
    return len(get_hits(df, target_id))


def print_metrics(results_csv, law_id, gt_count, k=50, recall_target=0.75):
    # Load once
    df = pd.read_csv(results_csv)

    # Compute
    recall = recall_at_k(df, law_id, gt_count, k)
    depth_full = search_depth_for_recall(df, law_id, gt_count, recall_target)
    best = first_hit_rank(df, law_id)
    
    print(f"Recall@{k}: {recall:.1%}")     
    print(f"Depth for {recall_target:.0%}: {depth_full}")    
    print(f"First Hit: {best}")               
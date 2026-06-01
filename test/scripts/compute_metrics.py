import pandas as pd
import numpy as np

# --- 1. Helper to get relevant rows (The "Hits") ---
def get_hits(df, target_ids):
    """Returns the rows matching one or more Law IDs (prefix) or Article IDs (exact)."""
    df = df.copy()
    df['node_id'] = df['node_id'].astype(str)

    prefixes = tuple(f"{law_id}#" for law_id in target_ids)
    return df[df['node_id'].str.startswith(prefixes)]

# --- 2. The Metric Functions ---

def recall_at_k(df, target_ids, total_gt, k):
    """Returns the percentage of the law found in the top K results."""
    hits = get_hits(df, target_ids)
    found_in_k = len(hits[hits['rank'] <= k])
    return found_in_k / total_gt

def search_depth_for_recall(df, target_ids, total_gt, recall_target=1.0):
    """
    Returns the Rank (depth) you must scroll to in order to find X% of the law.
    Returns -1 if that recall target is not reached in the file.
    """
    hits = get_hits(df, target_ids).sort_values('rank')
    
    # How many articles constitute X%? (e.g. 0.75 * 12 = 9 articles)
    needed_count = int(np.ceil(recall_target * total_gt))
    
    if len(hits) < needed_count:
        return -1 # Target not reachable with current results
        
    # Return the rank of the Nth item (iloc is 0-indexed)
    return hits.iloc[needed_count - 1]['rank']

def first_hit_rank(df, target_ids):
    """Returns the rank of the very first appearance, ignoring targets with no matches."""
    first_hits = []
    
    for target in target_ids:
        hits = get_hits(df, [target])
        if not hits.empty:
            first_hit = hits.sort_values('rank').iloc[0]['rank']
            first_hits.append(first_hit)
            
    if not first_hits:
        return -1  # Or float('nan') if no targets were found at all
        
    return float(np.mean(first_hits))

def total_recovered(df, target_ids):
    """Returns simply how many relevant items are in the file."""
    return len(get_hits(df, target_ids))


def print_metrics(results_csv, target_ids, gt_count, k=50, recall_target=0.75, output_path=None):
    # Load once
    df = pd.read_csv(results_csv)

    # Compute
    recall = recall_at_k(df, target_ids, gt_count, k)
    depth_full = search_depth_for_recall(df, target_ids, gt_count, recall_target)
    best = first_hit_rank(df, target_ids)

    lines = [
        f"Number of GT articles: {gt_count}",
        f"Recall@{k}: {recall:.1%}",
        f"Depth for {recall_target:.0%}: {depth_full}",
        f"Average First Hit: {best}",
    ]
    
    for line in lines:
        print(line)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    print_metrics(
        results_csv="test/test_outputs/golden_power/test3_results_golden_power.csv",
        target_ids=['2020|23', '2017|148', '2012|21'],
        gt_count=17,
        k=50,
        recall_target=0.75
    )      
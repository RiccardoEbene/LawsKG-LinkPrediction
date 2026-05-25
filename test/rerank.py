import json

import pandas as pd
from sentence_transformers import CrossEncoder
from test_utils import prepare_evaluation_prompt

MAX_ARTICLES = 50

def rerank_search_results(input_path: str, output_path: str, query: str, model_name: str = "BAAI/bge-reranker-v2-m3") -> pd.DataFrame:
    """
    Reads a CSV file containing search results, reranks them against a query 
    using a Cross-Encoder, and returns an updated DataFrame.
    """

    df = pd.read_csv(input_path)
    df["node_id"] = df["node_id"].astype(str).str.strip('"')
    df = df.head(MAX_ARTICLES)
    
    # Initialize the Cross-Encoder model
    model = CrossEncoder(model_name)
    
    enriched_texts = []
    for _, row in df.iterrows():
        context_prompt = prepare_evaluation_prompt(query, row['node_id'], "bolt://localhost:23034", ("", ""), llm_judge=False)
        enriched_texts.append(context_prompt)
    
    # Query-text pairs for the Cross-Encoder
    pairs = [[query, str(text)] for text in enriched_texts]
    
    scores = model.predict(pairs, batch_size=2)
    
    # Assign the new scores to the DataFrame
    df['rerank_score'] = scores
    
    # Sort by the new Cross-Encoder score
    df_reranked = df.sort_values(by='rerank_score', ascending=False).reset_index(drop=True)
    
    # Update the 'rank' column to reflect the new reranked order
    df_reranked['rank'] = df_reranked.index + 1
    
    # Save the reranked DataFrame to a new CSV file
    df_reranked.to_csv(output_path, index=False)
    
    return df_reranked


import pandas as pd

def score_ground_truth_positions(results_path: str, ground_truth_laws: list[str], max_articles: int = 100) -> dict:
    # 1. Load and process results
    df = pd.read_csv(results_path)[:max_articles]
    df["node_id"] = df["node_id"].astype(str).str.strip('"')
    df["law_id"] = df["node_id"].str.split("#").str[0]

    # 2. Filter out ground truth hits and sort them by rank
    gt_df = df[df["law_id"].isin(ground_truth_laws)].copy()
    gt_df = gt_df.sort_values(by="rank")
    ranks = gt_df["rank"].tolist()
    
    total_gt_available = len(ground_truth_laws)

    # Average Precision calculation
    if ranks and total_gt_available > 0:
        ap_sum = sum((idx + 1) / rank for idx, rank in enumerate(ranks))
        ap = ap_sum / total_gt_available
    else:
        ap = 0.0

    # Mean Reciprocal Rank calculation
    mrr = 1.0 / ranks[0] if ranks else 0.0
    
    # Hit Counts
    hits_at_1 = sum(rank <= 1 for rank in ranks)
    hits_at_5 = sum(rank <= 5 for rank in ranks)
    hits_at_10 = sum(rank <= 10 for rank in ranks)

    return {
        "gt_found_count": len(ranks),
        "gt_total_expected": total_gt_available,
        "ranks": ranks,
        "mean_rank": sum(ranks) / len(ranks) if ranks else 0,
        "median_rank": ranks[len(ranks) // 2] if ranks else 0,
        "mrr": mrr,
        "average_precision": ap,  # This is the AP for this single query file
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
    }

if __name__ == "__main__":
    '''
    reranked_df = rerank_search_results(
        input_path="test/test_outputs/ozono/old_results_ozono.csv",
        output_path="test/test_outputs/ozono/reranked_results_ozono.csv",
        query="Normativa, informazioni e obblighi per chi produce, utilizza, detiene le sostanze ozono lesive",
    )
    '''
    gt_laws = ["2017|51","2005|66","2005|128","2006|152","2007|205","2014|112"]
    evaluation_metrics = score_ground_truth_positions("test/test_outputs/ozono/test2_results_ozono.csv", gt_laws)
    with open("test/test_outputs/ozono/test2_ozono_gtMetrics.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_metrics, f, ensure_ascii=False, indent=2)
    print(evaluation_metrics)
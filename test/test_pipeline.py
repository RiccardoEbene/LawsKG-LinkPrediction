import os
import time

from scripts.insert_new_links import insert_new_links
from scripts.embedding_compute import compute_and_save_embeddings
from scripts.set_new_embeddings import update_embeddings_in_db
from scripts.query_search import perform_vector_search
from scripts.compute_metrics import print_metrics

# --- Configuration ---
CONFIG = {
    "URI": "bolt://localhost:23034",
    "AUTH": ("", ""),
    
    # File Paths
    "INPUT_LINKS": "output/NEW_pairs_nucleare_ranked.csv",
    "EMBEDDING_NPY": "test/test_outputs/nucleare_2010_31/DELETE_embeddings_dict_2010_31.npy",
    "SEARCH_RESULTS": "test/test_outputs/nucleare_2010_31/DELETE_results_nucleare.csv",
    
    # Experiment Parameters
    "QUERY": "Normativa sul nucleare e sulla gestione dei rifiuti radioattivi",
    "YEAR_FILTER": 1997,
    "LAW_ID_TARGET": "2010|31",
    "GT_TOTAL": 35, # Ground truth total articles
    "K_RECALL": 50
}

def main():
    print(">>> STEP 1: Inserting New Links")
    insert_new_links(
        new_links=CONFIG["INPUT_LINKS"], 
        uri=CONFIG["URI"], 
        k=1000, 
        wrong=False
    )
    
    print("\n>>> STEP 2: Computing Embeddings")
    compute_and_save_embeddings(
        input_edges_path=CONFIG["INPUT_LINKS"],
        output_npy_path=CONFIG["EMBEDDING_NPY"],
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\n>>> STEP 3: Updating Graph with New Embeddings")
    update_embeddings_in_db(
        npy_path=CONFIG["EMBEDDING_NPY"],
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\n>>> STEP 4: Performing Vector Search")
    perform_vector_search(
        query_text=CONFIG["QUERY"],
        year=CONFIG["YEAR_FILTER"],
        output_csv_path=CONFIG["SEARCH_RESULTS"],
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\n>>> STEP 5: Computing Metrics")
    print_metrics(
        results_csv=CONFIG["SEARCH_RESULTS"],
        law_id=CONFIG["LAW_ID_TARGET"],
        gt_count=CONFIG["GT_TOTAL"],
        k=CONFIG["K_RECALL"]
    )
    
    print("\n>>> PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
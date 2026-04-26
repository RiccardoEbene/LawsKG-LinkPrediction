import os
import time

from scripts.insert_new_links import insert_new_links
from scripts.embedding_compute import compute_and_save_embeddings_test1
from scripts.set_new_embeddings import update_embeddings_in_db
from scripts.query_search import perform_vector_search
from scripts.compute_metrics import print_metrics
from scripts.clean_up import clean_up

# --- Configuration ---
CONFIG = {
    "URI": "bolt://localhost:23034",
    "AUTH": ("", ""),

    "USE_OLD_EMBEDDINGS": False, # Set to True to run the pipeline with old embeddings for comparison
    
    # File Paths
    "INPUT_LINKS": "output/inference_test1/pairs_sostanze_ranked.csv",
    "SEARCH_RESULTS": "test/test_outputs/sostanze/new_results_sostanze.csv",
    "OLD_SEARCH_RESULTS": "test/test_outputs/sostanze/old_results_sostanze.csv",
    # Experiment Parameters
    "LINKS_TO_INSERT": 1000, # Number of new links to insert for the experiment
    "QUERY": "Normativa per il trattamento delle sostanze chimiche (regolamento REACH).",
    "YEAR_FILTER": 1997,
    "LAW_ID_TARGET": "2008|145",
    "GT_TOTAL": 14, # Ground truth total articles
    "K_RECALL": 50
}

def run_new_embeddings_pipeline():
    print("STEP 1: Inserting New Links")
    insert_new_links(
        new_links=CONFIG["INPUT_LINKS"], 
        uri=CONFIG["URI"], 
        k=CONFIG["LINKS_TO_INSERT"], 
        wrong=False
    )
    
    print("\nSTEP 2: Computing Embeddings")
    updated_nodes, new_embeddings = compute_and_save_embeddings_test1(
        input_edges_path=CONFIG["INPUT_LINKS"],
        n_inserted_links=CONFIG["LINKS_TO_INSERT"],
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\nSTEP 3: Updating Graph with New Embeddings")
    update_embeddings_in_db(
        embeddings_to_save=new_embeddings,
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\nSTEP 4: Performing Vector Search")
    perform_vector_search(
        query_text=CONFIG["QUERY"],
        year=CONFIG["YEAR_FILTER"],
        output_csv_path=CONFIG["SEARCH_RESULTS"],
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\nSTEP 5: Computing Metrics")
    print_metrics(
        results_csv=CONFIG["SEARCH_RESULTS"],
        law_id=CONFIG["LAW_ID_TARGET"],
        gt_count=CONFIG["GT_TOTAL"],
        k=CONFIG["K_RECALL"]
    )

    print("\nSTEP 6: Deleting new links and restoring old embeddings")
    clean_up(
        updated_nodes=updated_nodes, 
        driver_uri=CONFIG["URI"], 
        auth=CONFIG["AUTH"]
    )

    print("\nPIPELINE COMPLETE")


def run_old_embeddings_pipeline():
    print("\nSTEP 1: Performing Vector Search on old embeddings")
    perform_vector_search(
        query_text=CONFIG["QUERY"],
        year=CONFIG["YEAR_FILTER"],
        output_csv_path=CONFIG["OLD_SEARCH_RESULTS"],
        use_old_embeddings=True,
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\nSTEP 2: Computing Metrics")
    print_metrics(
        results_csv=CONFIG["OLD_SEARCH_RESULTS"],
        law_id=CONFIG["LAW_ID_TARGET"],
        gt_count=CONFIG["GT_TOTAL"],
        k=CONFIG["K_RECALL"]
    )

    print("\nOLD EMBEDDINGS PIPELINE COMPLETE")


def main(use_old_embeddings):
    if use_old_embeddings:
        run_old_embeddings_pipeline()
    else:
        run_new_embeddings_pipeline()

if __name__ == "__main__":
    main(use_old_embeddings=CONFIG["USE_OLD_EMBEDDINGS"])
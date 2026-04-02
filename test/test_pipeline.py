import os
import time

from scripts.insert_new_links import insert_new_links
from scripts.embedding_compute import compute_and_save_embeddings
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
    "INPUT_LINKS": "output/pairs_ozono_ranked_full_train.csv",
    "EMBEDDING_NPY": "test/test_outputs/ozono/new_embeddings_dict_1993_549_train.npy",
    "SEARCH_RESULTS": "test/test_outputs/ozono/new_results_ozono_train.csv",
    "OLD_SEARCH_RESULTS": "test/test_outputs/ozono/old_results_ozono.csv",
    "OLD_EMBEDDING_NPY": "test/test_outputs/ozono/old_embeddings_1993_549.npy",
    # Experiment Parameters
    "QUERY": "Normativa, informazioni e obblighi per chi produce, utilizza, detiene le sostanze ozono lesive",
    "YEAR_FILTER": 1993,
    "LAW_ID_TARGET": "1993|549",
    "GT_TOTAL": 17, # Ground truth total articles
    "K_RECALL": 50
}

def run_new_embeddings_pipeline():
    print("STEP 1: Inserting New Links")
    insert_new_links(
        new_links=CONFIG["INPUT_LINKS"], 
        uri=CONFIG["URI"], 
        k=1000, 
        wrong=False
    )
    
    print("\nSTEP 2: Computing Embeddings")
    compute_and_save_embeddings(
        input_edges_path=CONFIG["INPUT_LINKS"],
        output_npy_path=CONFIG["EMBEDDING_NPY"],
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\nSTEP 3: Updating Graph with New Embeddings")
    update_embeddings_in_db(
        npy_path=CONFIG["EMBEDDING_NPY"],
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

    print("\nSTEP 6: Deleting new link and restoring old embeddings")
    clean_up(
        old_emb=CONFIG["OLD_EMBEDDING_NPY"], 
        driver_uri=CONFIG["URI"], 
        auth=CONFIG["AUTH"]
    )

    print("\nPIPELINE COMPLETE")


def run_old_embeddings_pipeline():
    print("STEP 1: Restoring Old Embeddings in Graph")
    update_embeddings_in_db(
        npy_path=CONFIG["OLD_EMBEDDING_NPY"],
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\nSTEP 2: Performing Vector Search")
    perform_vector_search(
        query_text=CONFIG["QUERY"],
        year=CONFIG["YEAR_FILTER"],
        output_csv_path=CONFIG["OLD_SEARCH_RESULTS"],
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\nSTEP 3: Computing Metrics")
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
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
    
    # File Paths
    "INPUT_LINKS": "output/inference_test2/pairs_ozono_ranked_2.csv",
    "SEARCH_RESULTS": "test/test_outputs/ozono/test2_results_ozono.csv",
    # Experiment Parameters
    "LINKS_TO_INSERT": 1000, # Number of new links to insert for the experiment
    "QUERY": "Normativa, informazioni e obblighi per chi produce, utilizza, detiene le sostanze ozono lesive",
    "YEAR_FILTER": 1993,
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
    updated_nodes, new_embeddings = compute_and_save_embeddings(
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

    print("\nSTEP 5: Deleting new links and restoring old embeddings")
    clean_up(
        updated_nodes=updated_nodes, 
        driver_uri=CONFIG["URI"], 
        auth=CONFIG["AUTH"]
    )

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    run_new_embeddings_pipeline()
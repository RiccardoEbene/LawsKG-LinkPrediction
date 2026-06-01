from test.scripts.insert_new_links import insert_new_links
from test.scripts.embedding_compute import compute_and_save_embeddings_test1
from test.scripts.set_new_embeddings import update_embeddings_in_db
from test.scripts.query_search import perform_vector_search
from test.scripts.compute_metrics import print_metrics
from test.scripts.clean_up import clean_up
from lists_dataset import get_law_data

# --- Configuration ---
CONFIG = {
    "URI": "bolt://localhost:23034",
    "AUTH": ("", ""),

    "USE_OLD_EMBEDDINGS": False, # Set to True to run the pipeline with old embeddings for comparison
    "MOST_CITED": False, # Set to False to use the second set of chosen laws for each topic (less cited ones)

    # File Paths
    "INPUT_LINKS_TEMPLATE": "output/inference_test4/pairs_{topic}_ranked.csv",
    "SEARCH_RESULTS_TEMPLATE": "test/test_outputs/{topic}/test4_results_{topic}.csv",
    "METRICS_OUTPUT_TEMPLATE": "test/test_outputs/{topic}/test4_{topic}_metrics",
    "OLD_SEARCH_RESULTS_TEMPLATE": "test/test_outputs/{topic}/old_results_{topic}.csv",
    "OLD_METRICS_OUTPUT_TEMPLATE": "test/test_outputs/{topic}/old_test4_{topic}_metrics",

    # Experiment Parameters
    "LINKS_TO_INSERT": 1000, # Number of new links to insert for the experiment
    # "QUERY": "Normativa per il trattamento delle sostanze chimiche (regolamento REACH).",
    # "YEAR_FILTER": 1997,
    # "GT_TOTAL": 14, # Ground truth total articles
    # "TARGET_LAWS": ["2008|145"],
    "K_RECALL": 50
}

def get_topic_configs():
    years, texts, laws, topics, chosen_laws, gt_counts = get_law_data(most_cited=CONFIG["MOST_CITED"])

    return [
        {
            "topic": topic,
            "text": text,
            "year": year,
            "chosen_laws": target_laws,
            "laws": law_ids,
            "gt_count": gt_count
        }
        for topic, text, year, target_laws, law_ids, gt_count in zip(topics, texts, years, chosen_laws, laws, gt_counts)
    ]


def run_new_embeddings_pipeline(topic_name, topic_text, topic_year, target_laws, target_gt_count):
    input_links = CONFIG["INPUT_LINKS_TEMPLATE"].format(topic=topic_name)
    search_results = CONFIG["SEARCH_RESULTS_TEMPLATE"].format(topic=topic_name)
    metrics_output = CONFIG["METRICS_OUTPUT_TEMPLATE"].format(topic=topic_name)

    print(f"\n=== Topic: {topic_name} ===")
    print(f"Query text: {topic_text}")
    print(f"Year filter: {topic_year}")
    print(f"Target laws: {target_laws}")

    print("STEP 1: Inserting New Links")
    insert_new_links(
        new_links=input_links, 
        uri=CONFIG["URI"], 
        k=CONFIG["LINKS_TO_INSERT"], 
        wrong=False
    )
    
    print("\nSTEP 2: Computing Embeddings")
    updated_nodes, new_embeddings = compute_and_save_embeddings_test1(
        input_edges_path=input_links,
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
        query_text=topic_text,
        year=topic_year,
        output_csv_path=search_results,
        driver_uri=CONFIG["URI"],
        auth=CONFIG["AUTH"]
    )

    print("\nSTEP 5: Computing Metrics")
    print_metrics(
        results_csv=search_results,
        target_ids=target_laws,
        gt_count=target_gt_count,
        k=CONFIG["K_RECALL"],
        output_path=metrics_output,
    )

    print("\nSTEP 6: Deleting new links and restoring old embeddings")
    clean_up(
        updated_nodes=updated_nodes, 
        driver_uri=CONFIG["URI"], 
        auth=CONFIG["AUTH"]
    )

    print("\nPIPELINE COMPLETE")


def run_all_new_embeddings_pipelines():
    for topic_config in get_topic_configs():
        run_new_embeddings_pipeline(
            topic_name=topic_config["topic"],
            topic_text=topic_config["text"],
            topic_year=topic_config["year"],
            target_laws=topic_config["chosen_laws"],
            target_gt_count=topic_config["gt_count"]
        )


def run_old_embeddings_pipeline():
    for topic_config in get_topic_configs():
        topic_name = topic_config["topic"]
        topic_text = topic_config["text"]
        topic_year = topic_config["year"]
        target_laws = topic_config["chosen_laws"]
        target_gt_count = topic_config["gt_count"]
        search_results = CONFIG["OLD_SEARCH_RESULTS_TEMPLATE"].format(topic=topic_name)
        metrics_output = CONFIG["OLD_METRICS_OUTPUT_TEMPLATE"].format(topic=topic_name)

        print(f"\n=== Topic: {topic_name} ===")
        print(f"Query text: {topic_text}")
        print(f"Year filter: {topic_year}")
        print(f"Target laws: {target_laws}")

        # print("STEP 1: Performing Vector Search on old embeddings")
        # perform_vector_search(
        #     query_text=topic_text,
        #     year=topic_year,
        #     output_csv_path=search_results,
        #     use_old_embeddings=True,
        #     driver_uri=CONFIG["URI"],
        #     auth=CONFIG["AUTH"]
        # )

        print("\nSTEP 2: Computing Metrics")
        print_metrics(
            results_csv=search_results,
            target_ids=target_laws,
            gt_count=target_gt_count,
            k=CONFIG["K_RECALL"],
            output_path=metrics_output
        )

    print("\nOLD EMBEDDINGS PIPELINE COMPLETE")


def main(use_old_embeddings):
    if use_old_embeddings:
        run_old_embeddings_pipeline()
    else:
        run_all_new_embeddings_pipelines()

if __name__ == "__main__":
    main(use_old_embeddings=CONFIG["USE_OLD_EMBEDDINGS"])
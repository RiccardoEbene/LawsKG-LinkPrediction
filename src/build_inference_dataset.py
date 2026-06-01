import pandas as pd
from itertools import combinations
from neo4j import GraphDatabase
from src.inference import inference
from src.utils import delete_invalid_couples
from lists_dataset import get_law_data

def get_nodes_after_year(year: int, uri: str = "bolt://localhost:23034"):
    driver = GraphDatabase.driver(uri, auth=("", ""))
    
    query = """
        MATCH (l:NationalLaw)-[r]->(a:LawUnit{_flag_remove:false})
        WHERE (type(r) = "HAS_ARTICLE" OR type(r) = "HAS_ATTACHMENT") AND
            l.publicationDate > localDateTime({year:$year, month:1, day:1}) 
        RETURN DISTINCT a.id as node_id
        """
    
    with driver.session() as session:
        result = session.run(query, year=year)
        records = [record.data() for record in result]
    
    driver.close()

    df = pd.DataFrame(records)
    df = df.astype({"node_id": str})
    # df.to_csv(f"output/nodes_after_{year}.csv", index=False)

    return df

def get_unit_from_law(law_id: str, uri: str = "bolt://localhost:23034"):
    driver = GraphDatabase.driver(uri, auth=("", ""))
    
    query = """
        MATCH (l:Law {id:$law_id})-[r]->(a:LawUnit{_flag_remove:false})
        WHERE type(r) = "HAS_ARTICLE" OR type(r) = "HAS_ATTACHMENT"
        RETURN DISTINCT a.id as node_id
        """
    
    with driver.session() as session:
        result = session.run(query, law_id=law_id)
        records = [record.data() for record in result]
    
    driver.close()

    df = pd.DataFrame(records)
    df = df.astype({"node_id": str})

    return df

import pandas as pd


def get_all_units_from_k_laws(law_ids: list) -> pd.DataFrame:
    """Calls get_unit_from_law for a list of law_ids and combines them

    into a single DataFrame.
    """
    all_dfs = []

    for law_id in law_ids:
        df_law = get_unit_from_law(law_id)

        # Only proceed if the dataframe isn't empty
        if not df_law.empty:
            all_dfs.append(df_law)

    # Combine all individual dataframes into one
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        # Return an empty DataFrame with the correct schema if no data was found
        combined_df = pd.DataFrame(columns=["node_id"])

    return combined_df

# combine two dataframes of node ids into couples
def build_inference_set(df1, df2):
    pairs = []
    for id1 in df1['node_id']:
        for id2 in df2['node_id']:
            pairs.append({'node_1': id1, 'node_2': id2})
    pairs_df = pd.DataFrame(pairs)
    # pairs_df.to_csv(output_csv, index=False)

    return pairs_df


def has_existing_connection(main_df: pd.DataFrame, exclude_csv: str = "data/connected_nodes.csv") -> pd.DataFrame:
    # Read the CSV containing the couples to exclude
    exclude_df = pd.read_csv(exclude_csv)
    
    # Perform a left merge to identify which rows from main_df exist in exclude_df
    merged_df = main_df.merge(exclude_df, on=['node_1', 'node_2'], how='left', indicator=True)

    filtered_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
    dropped_count = len(main_df) - len(filtered_df)
    print(f"Dropped {dropped_count} couples already present in {exclude_csv}.")

    # Keep only the rows that are uniquely in the left dataframe (main_df) and drop the indicator column
    return filtered_df

def build_inference_dataset_from_law(

        year: int, 
        law: str, 
        output_csv: str, 
        nodes_csv: str,
        uri: str = "bolt://localhost:23034", 
):
    df1 = get_unit_from_law(law, uri)
    df2 = get_nodes_after_year(year, uri)
    df2 = df2[~df2['node_id'].isin(df1['node_id'])]
    nodes_df = pd.read_csv(nodes_csv, dtype={'node_id': str})

    # combine to get inference pairs
    inference_pairs_df = build_inference_set(df1, df2)

    print("Inference dataset built, deleting invalid couples...")
    inference_pairs_df = has_existing_connection(inference_pairs_df)

    # filter out couples including unknown nodes
    inference_pairs_df = delete_invalid_couples(inference_pairs_df, nodes_df)

    # shuffle
    inference_pairs_df = inference_pairs_df.sample(frac=1).reset_index(drop=True)
    inference_pairs_df.to_csv(output_csv, index=False)

    print(len(inference_pairs_df), "inference pairs generated.")
    print(f"Inference pairs saved to {output_csv}")


def get_topk_nodes(input_file: str, k: int):
    df = pd.read_csv(input_file, dtype={"node_id": str})
    topk_df = df.head(k)
    return topk_df[["node_id"]]

def build_inference_dataset_from_topk(
    year: int,
    input_file: str,
    output_csv: str,
    nodes_csv: str,
    k: int = 1000,
    uri: str = "bolt://localhost:23034",
):
    df1 = get_topk_nodes(input_file, k)
    df2 = get_nodes_after_year(year, uri)
    df2 = df2[~df2['node_id'].isin(df1['node_id'])]
    nodes_df = pd.read_csv(nodes_csv, dtype={'node_id': str})

    # Compute cartesian product to get inference pairs
    inference_pairs_df = build_inference_set(df1, df2)

    print("Inference dataset built, deleting invalid couples...")
    # Filter out pairs that already have a connection in the graph
    inference_pairs_df = has_existing_connection(inference_pairs_df)

    # Filter out pairs that include unknown nodes
    inference_pairs_df = delete_invalid_couples(inference_pairs_df, nodes_df)

    # shuffle
    inference_pairs_df = inference_pairs_df.sample(frac=1).reset_index(drop=True)

    inference_pairs_df.to_csv(output_csv, index=False)

    print(len(inference_pairs_df), "inference pairs generated.")
    print(f"Inference pairs saved to {output_csv}")

def get_most_cited_laws(law_ids: list, k: int, uri: str = "bolt://localhost:23034") -> list:
    """ Given a list of law_ids, query the database to get their number of citations, sort them by citations and return the top k law_ids. """
    driver = GraphDatabase.driver(uri, auth=("", ""))

    query = """
        MATCH (l:Law)
        WHERE l.id IN $law_ids
        RETURN l.id as law_id, l.numberOfCitations as citations, l.numberOfArticles as articles, l.numberOfAttachment as attachments
        """

    with driver.session() as session:
        result = session.run(query, law_ids=law_ids)
        records = [record.data() for record in result]

    driver.close()

    # Drop rows where number of articles is above 100
    records = [r for r in records if (r["articles"]+r["attachments"]) <= 100]

    # Sort the list of dictionaries by citations descending and slice top k
    records.sort(key=lambda x: x["citations"], reverse=False) # Ora sono le meno citate
    return [r["law_id"] for r in records[:k]]

def build_inference_datasets_from_k_laws(
    nodes_csv: str,  
    uri: str = "bolt://localhost:23034",
    base_output_csv: str = "data/inference_test3",
    k: int = 3,
    most_cited: bool = True
):
    """ Build inference dataset for each topic, selecting top k ground truth laws by citations and combining their articles with all articles after reference year """
    
    # Get ground-truth dataset
    years, texts, laws, topics, _, _ = get_law_data(most_cited=most_cited)

    for i, topic in enumerate(topics):
        output_csv = f"{base_output_csv}/inference_pairs_{topic}.csv"

        # Pick the top k laws for the current topic for number of citations
        chosen_laws = get_most_cited_laws(laws[i], k)

        print(f"Topic: {topic} - Selected top {k} laws: {chosen_laws}")

        # Get units from the top k laws for the current topic
        df1 = get_all_units_from_k_laws(chosen_laws)

        # Prevent explosion of the inference dataset by checking the number of units in df1
        if len(df1) > 110:
            print(f"Warning: {len(df1)} units found for topic '{topic}' with top {k} laws. Consider reducing k to limit the size of the inference dataset.")
            continue

        df2 = get_nodes_after_year(years[i], uri)
        df2 = df2[~df2['node_id'].isin(df1['node_id'])]
        nodes_df = pd.read_csv(nodes_csv, dtype={'node_id': str})

        # Compute cartesian product to get inference pairs
        inference_pairs_df = build_inference_set(df1, df2)

        print("Inference dataset built, deleting invalid couples...")
        # Filter out pairs that already have a connection in the graph
        inference_pairs_df = has_existing_connection(inference_pairs_df)

        # Filter out pairs that include unknown nodes
        inference_pairs_df = delete_invalid_couples(inference_pairs_df, nodes_df)

        # shuffle
        inference_pairs_df = inference_pairs_df.sample(frac=1).reset_index(drop=True)

        inference_pairs_df.to_csv(output_csv, index=False)

        print(len(inference_pairs_df), "inference pairs generated.")
        print(f"Inference pairs saved to {output_csv}")

if __name__ == "__main__":
    URI = "bolt://localhost:23034"
    YEAR = 1993
    LAW = "1993|549"
    # OLD_CLOSEST = f"test/test_outputs/combustibili/old_results_combustibili.csv"
    NODES_CSV = f"data/all_nodes.csv"
    #OUTPUT_CSV = f"data/inference_test1/inference_pairs_ozono.csv"

    build_inference_datasets_from_k_laws(
        nodes_csv=NODES_CSV,
        uri=URI,
        base_output_csv="data/inference_test4",
        k=3
    )
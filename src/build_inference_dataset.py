import pandas as pd
from itertools import combinations
from neo4j import GraphDatabase
from src.inference import inference
from src.utils import delete_invalid_couples

def get_nodes_after_year(year: int, uri: str):
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

def get_unit_from_law(law_id: str, uri: str):
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


def get_nodes_between_years(start_year: int, end_year: int, uri: str):
    driver = GraphDatabase.driver(uri, auth=("", ""))

    query = """
        MATCH (l:NationalLaw)-[r]->(a:LawUnit {_flag_remove:false})
        WHERE (type(r) = "HAS_ARTICLE" OR type(r) = "HAS_ATTACHMENT") AND
            l.publicationDate >= localDateTime({year:$start_year, month:1, day:1}) AND
            l.publicationDate <= localDateTime({year:$end_year, month:12, day:31})
        RETURN DISTINCT a.id as node_id
        """

    with driver.session() as session:
        result = session.run(query, start_year=start_year, end_year=end_year)
        records = [record.data() for record in result]

    driver.close()

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["node_id"]) 

    df = df.astype({"node_id": str})
    return df

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
        uri: str, 
        year: int, 
        law: str, 
        output_csv: str, 
        nodes_csv: str
):
    df1 = get_unit_from_law(law, uri)
    df2 = get_nodes_after_year(year, uri)
    df2 = df2[~df2['node_id'].isin(df1['node_id'])]
    nodes_df = pd.read_csv(nodes_csv, dtype={'node_id': str})

    # combine to get inference pairs
    inference_pairs_df = build_inference_set(df1, df2)

    # filter out couples including unknown nodes
    inference_pairs_df = delete_invalid_couples(inference_pairs_df, nodes_df)

    inference_pairs_df.to_csv(output_csv, index=False)

    print(len(inference_pairs_df), "inference pairs generated.")
    print(f"Inference pairs saved to {output_csv}")


def build_inference_dataset_between_years(
    uri: str,
    start_year: int,
    end_year: int,
    output_csv: str,
    nodes_csv: str,
):
    lawunit_df = get_nodes_between_years(start_year, end_year, uri)
    node_ids = lawunit_df["node_id"].dropna().astype(str).unique().tolist()

    # Build unique undirected pairs only once (no self-pairs, no duplicates).
    pairs_df = pd.DataFrame(combinations(node_ids, 2), columns=["node_1", "node_2"])

    if pairs_df.empty:
        pairs_df.to_csv(output_csv, index=False)
        print("0 inference pairs generated.")
        print(f"Inference pairs saved to {output_csv}")
        return

    pairs_df = has_existing_connection(pairs_df)

    nodes_df = pd.read_csv(nodes_csv, dtype={"node_id": str})
    pairs_df = delete_invalid_couples(pairs_df, nodes_df)

    pairs_df.to_csv(output_csv, index=False)

    print(len(pairs_df), "inference pairs generated.")
    print(f"Inference pairs saved to {output_csv}")

def get_topk_nodes(input_file: str, k: int):
    df = pd.read_csv(input_file, dtype={"node_id": str})
    topk_df = df.head(k)
    return topk_df[["node_id"]]

def build_inference_dataset_from_topk(
    uri: str,
    year: int,
    input_file: str,
    output_csv: str,
    nodes_csv: str,
    k: int = 1000,
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

if __name__ == "__main__":
    URI = "bolt://localhost:23034"
    YEAR = 2005
    # LAW = "2008|145"
    OLD_CLOSEST = f"test/test_outputs/combustibili/old_results_combustibili.csv"
    NODES_CSV = f"data/all_nodes.csv"
    OUTPUT_CSV = f"data/inference_test2/inference_pairs_combustibili_2.csv"

    build_inference_dataset_from_topk(URI, YEAR, OLD_CLOSEST, OUTPUT_CSV, NODES_CSV, k=100)
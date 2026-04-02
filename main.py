import os
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from neo4j import GraphDatabase
from test.scripts.set_new_embeddings import update_embeddings_in_db

BOLT_URL = os.getenv("BOLT_MEMGRAPH", "bolt://localhost:23034")





def update_parquet_with_db_embeddings(parquet_file: str, emb_file: str):
    '''
    Update input parquet file with embedding from npy embedding file

    '''
    input_embeddings = np.load(emb_file, allow_pickle=True).item()
    df = pd.read_parquet(parquet_file)

    for id in input_embeddings.keys():
        id = str(id)
        embedding = [float(x) for x in input_embeddings[id]]
        
        # Check if embedding is valid
        if len(embedding) != 1024:
            print(f"Warning: Node {id} has embedding of size {len(embedding)}, skipping (expected 1024)")
            continue
        
        # Check for NaN or Inf values
        if any(np.isnan(x) or np.isinf(x) for x in embedding):
            print(f"Warning: Node {id} has NaN or Inf values in embedding, skipping")
            continue

        # Update id entry in df with new embedding
        df.loc[df['node_id'] == id, 'embedding'] = [embedding]

    df.to_parquet(parquet_file)
    print("Parquet file updated with new embeddings.")


# Print embedding of all articles of law X from parquet input file
def print_law_articles_embeddings(art_id: str, parquet_file: str):
    df1 = pd.read_parquet(parquet_file)
    print(df1[df1['node_id'] == art_id]['embedding'].values[:10])

# Get embeddings from parquet file and save them into npy dict
def get_embeddings(parquet_file: str, output_file: str, prefix: str = "1993|549#"):
    df = pd.read_parquet(parquet_file)

    mask = df['node_id'].astype(str).str.startswith(prefix)
    filtered_df = df[mask]
    
    emb_dict = {record['node_id']: record['embedding'] for record in filtered_df.to_dict('records')}
    print(emb_dict.keys())
    np.save(output_file, emb_dict)

# get count of rows in edges file where node_1 or node_2 is not in nodes parquet file
def count_rows_with_missing_nodes(edges_file: str = "data/all_edges.csv", nodes_file: str = "data/nodes.parquet") -> int:
    edges = pd.read_csv(edges_file)
    nodes = pd.read_parquet(nodes_file)["node_id"]
    return ((~edges["node_1"].isin(nodes)) | (~edges["node_2"].isin(nodes))).sum()

# Get list of unique node IDs in edges file that are not in nodes parquet file
def get_missing_node_ids(edges_file: str = "data/all_edges.csv", nodes_file: str = "data/nodes.parquet") -> list:
    edges = pd.read_csv(edges_file)
    nodes = pd.read_parquet(nodes_file)["node_id"]
    
    # Identify IDs in node_1 and node_2 that are not in the nodes set
    missing_n1 = edges.loc[~edges["node_1"].isin(nodes), "node_1"]
    missing_n2 = edges.loc[~edges["node_2"].isin(nodes), "node_2"]
    
    # Combine, drop duplicates, and return as a list
    return pd.concat([missing_n1, missing_n2]).unique().tolist()

def create_nodes_parquet_from_edges(
    edges_file: str = "data/edges.csv",
    all_nodes_file: str = "data/all_nodes_emb.parquet",
    nodes_csv_file: str = "data/nodes.csv",
    output_file: str = "data/nodes.parquet",
):
    # Read only required columns and deduplicate while preserving first-seen order.
    edges = pd.read_csv(edges_file, usecols=["node_1", "node_2"])
    unique_nodes = pd.unique(
        pd.concat(
            [edges["node_1"], edges["node_2"]],
            ignore_index=True,
        )
    )

    nodes_df = pd.DataFrame({"node_id": unique_nodes})
    nodes_df.to_csv(nodes_csv_file, index=False)

    # Filter parquet read to only needed IDs and columns for efficiency.
    node_ids = nodes_df["node_id"].tolist()
    nodes_with_emb = pd.read_parquet(
        all_nodes_file,
        columns=["node_id", "embedding"],
        filters=[("node_id", "in", node_ids)],
    )

    # Enforce uniqueness and keep the original nodes.csv order.
    nodes_with_emb = nodes_with_emb.drop_duplicates(subset=["node_id"], keep="first")
    nodes_with_emb = nodes_df.merge(nodes_with_emb, on="node_id", how="left")
    nodes_with_emb.to_parquet(output_file, index=False)

    print(f"Unique nodes found: {len(nodes_df)}")
    print(f"Nodes with embedding written: {len(nodes_with_emb)}")


def count_parquet_node(parquet_file: str = "data/nodes.parquet") -> int:
    df = pd.read_parquet(parquet_file)
    return len(df["node_id"].unique())

def get_num_articles(numpy_file: str) -> int:
    emb_dict = np.load(numpy_file, allow_pickle=True).item()
    return len(emb_dict)

if __name__ == "__main__":
    df_1 = pd.read_parquet("data/train_nodes.parquet")
    print(len(df_1))



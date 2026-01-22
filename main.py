import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from test.scripts.set_new_embeddings import update_embeddings_in_db

BOLT_URL = os.getenv("BOLT_MEMGRAPH", "bolt://localhost:23034")

'''
Update input parquet file with embedding from npy embedding file
'''
def update_parquet_with_db_embeddings(parquet_file: str, emb_file: str):
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

if __name__ == "__main__":
    update_parquet_with_db_embeddings("data/nodes.parquet", "test/test_outputs/combustibili_2017_51/old_embeddings_dict_2017_51.npy")
    
    # print_law_articles_embeddings("2015|150#2", "data/all_nodes_emb.parquet")
    






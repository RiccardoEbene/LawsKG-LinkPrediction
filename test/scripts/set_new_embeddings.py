import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import os

def update_embeddings_in_db(npy_path, driver_uri, auth):
    """Updates node embeddings in the Memgraph database from a .npy file."""
    driver = GraphDatabase.driver(driver_uri, auth=auth)

    input_embeddings = np.load(npy_path, allow_pickle=True).item()

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
        
        driver.execute_query("""
        MATCH (n:LawUnit)
        WHERE n.id = $node_id
        SET n.embedding = $embedding
        """, node_id=id, embedding=embedding)

    print("Update complete.")
    driver.close()
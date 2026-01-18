import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase


BOLT_URL = os.getenv("BOLT_MEMGRAPH", "bolt://localhost:23034")

input_edges = "output/pairs_pesticidi_ranked.csv"

df = pd.read_csv(input_edges)

nodes_1 = set(df['node_1'].tolist())
print(len(nodes_1), "unique nodes in node_1")

# extract embedding of nodes in nodes_1
driver = GraphDatabase.driver(BOLT_URL, auth=("", ""))
embeddings, _, _ = driver.execute_query(f"""
MATCH (n)
WHERE n.id IN $node_ids
RETURN n.id AS id, n.embedding AS embedding
""", node_ids=list(nodes_1))    

emb_dict = {record['id']: record['embedding'] for record in embeddings}
# save in a file
np.save("vector_search/pesticidi_2012_150/old_embeddings_dict_2012_150.npy", emb_dict)

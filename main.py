import memgraph_util
import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase

BOLT_URL = os.getenv("BOLT_MEMGRAPH", "bolt://localhost:23034")


# return number of rows in the parquet file
def count_parquet_rows(parquet_file):
    table = pd.read_parquet(parquet_file, columns=[])
    return len(table)

if __name__ == "__main__":
    input_file = "data/all_nodes_emb.parquet"
    print(count_parquet_rows(input_file))
    df = pd.read_parquet(input_file)
    print(list(df[df['node_id']=='1993|549#4']['embedding'].iloc[0][-10:])) 
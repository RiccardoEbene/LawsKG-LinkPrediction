import pandas as pd
import numpy as np
from neo4j import GraphDatabase

def get_in_notes_edges(uri: str):
    driver = GraphDatabase.driver(uri, auth=("", ""))
    
    query = """
        MATCH (a:LawUnit{_flag_remove:false})-[r:IN_NOTES]->(b:LawUnit{_flag_remove:false})
        RETURN a.id as node_1, b.id as node_2, 1 AS label
        """
    
    with driver.session() as session:
        result = session.run(query)
        records = [record.data() for record in result]
    
    driver.close()

    df = pd.DataFrame(records)

    print(f"Retrieved {len(df)} positive edges from IN_NOTES relationship.")

    return df

def get_random_negative_edges(uri: str, num_samples: int):
    driver = GraphDatabase.driver(uri, auth=("", ""))
    
    query = """
    MATCH (l:Law)-[:HAS_ARTICLE|HAS_ATTACHMENT]->(n:LawUnit{_flag_remove:false})
    WHERE l.publicationDate >= localDateTime({year:1985, month:1, day:1})

    WITH n, rand() AS r1
    ORDER BY r1
    LIMIT 5000

    WITH collect(n) AS node_pool
    UNWIND node_pool AS a
    UNWIND node_pool AS b
    WITH a, b
    WHERE id(a) < id(b)

    OPTIONAL MATCH (a)-[e]-(b)
    WITH a, b, e
    WHERE e IS NULL

    WITH a, b, rand() AS r2
    ORDER BY r2
    LIMIT $num_samples

    RETURN a.id as node_1, b.id as node_2, 0 as label;
        """
    
    print(f"Retrieving {num_samples} random negative edges...")

    with driver.session() as session:
        result = session.run(query, num_samples=num_samples)
        records = [record.data() for record in result]
    
    driver.close()

    df = pd.DataFrame(records)

    print(f"Retrieved {len(df)} random negative edges.")

    return df

def get_hard_negative_edges(uri: str, num_samples: int):
    driver = GraphDatabase.driver(uri, auth=("", ""))
    
    query = """
        MATCH (l:Law)-[:HAS_ARTICLE|HAS_ATTACHMENT]->(a:LawUnit{_flag_remove:false})
        WHERE l.publicationDate >= localDateTime({year:1985, month:1, day:1})
        WITH a
        MATCH (a)-[:CITES]->(:LawUnit)-[:CITES]->(b:LawUnit{_flag_remove:false})
        WHERE a.id <> b.id
        AND NOT EXISTS((a)-[]-(b))
        RETURN a.id AS node_1, b.id AS node_2, 0 AS label
        ORDER BY rand()
        LIMIT $num_samples;
        """
    
    with driver.session() as session:
        result = session.run(query, num_samples=num_samples)
        records = [record.data() for record in result]
    
    driver.close()

    df = pd.DataFrame(records)
    print(f"Retrieved {len(df)} hard negative edges.")
    return df

def filter_edges(df, nodes_path):
    nodes = set(pd.read_csv(nodes_path)['node_id'].astype(str).str.strip('"'))
    initial_count = len(df)
    df = df[df['node_1'].isin(nodes) & df['node_2'].isin(nodes)].copy()
    dropped_missing_nodes = initial_count - len(df)
    print(f"Dropped couples due to missing nodes in nodes_path: {dropped_missing_nodes}")


    before_dedup_count = len(df)
    df['sorted_nodes'] = df.apply(lambda row: tuple(sorted([row['node_1'], row['node_2']])), axis=1)
    df = df.drop_duplicates(subset='sorted_nodes').drop(columns='sorted_nodes')
    dropped_duplicates = before_dedup_count - len(df)
    print(f"Dropped duplicate couples (order-insensitive): {dropped_duplicates}")

    return df

if __name__ == "__main__":
    uri = "bolt://localhost:23034"

    num_positives = 50586
    tot_negatives = num_positives * 3
    num_random = int(tot_negatives * 0.7)
    num_hard = tot_negatives - num_random
    
    positive_df = get_in_notes_edges(uri)
    random_negatives_df = get_random_negative_edges(uri, num_samples=num_random)
    hard_negatives_df = get_hard_negative_edges(uri, num_samples=num_hard)

    # check the random negative
    random_negatives_df.to_csv("data/random_negatives_samples.csv", index=False)

    all_edges_df = pd.concat([positive_df, random_negatives_df, hard_negatives_df], ignore_index=True)

    all_edges_df = filter_edges(all_edges_df, "data/train_nodes.csv")
    all_edges_df = all_edges_df.sample(frac=1, random_state=42).reset_index(drop=True)

    all_edges_df.to_csv("data/in_notes_set_labeled_3.csv", index=False)
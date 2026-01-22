import torch
import pandas as pd
import dgl
from neo4j import GraphDatabase
from src.inference import inference
from src.utils import delete_invalid_couples

def get_nodes_after_year(year: int, uri: str):
    driver = GraphDatabase.driver(uri, auth=("", ""))
    
    query = """
        MATCH (l:Law)-[r]->(a:LawUnit)
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
        MATCH (l:Law {id:$law_id})-[r]->(a:LawUnit)
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

# combine two dataframes of node ids into couples
def build_inference_set(df1, df2):
    pairs = []
    for id1 in df1['node_id']:
        for id2 in df2['node_id']:
            pairs.append({'node_1': id1, 'node_2': id2})
    pairs_df = pd.DataFrame(pairs)
    # pairs_df.to_csv(output_csv, index=False)

    return pairs_df

if __name__ == "__main__":
    URI = "bolt://localhost:23034"
    YEAR = 1993
    LAW = "1993|549"
    OUTPUT_CSV = f"data/inference_pairs_ozono_all.csv"
    nodes_csv = f"data/all_nodes.csv"

    df1 = get_unit_from_law(LAW, URI)
    df2 = get_nodes_after_year(YEAR, URI)
    df2 = df2[~df2['node_id'].isin(df1['node_id'])]
    nodes_df = pd.read_csv(nodes_csv, dtype={'node_id': str})

    # combine to get inference pairs
    inference_pairs_df = build_inference_set(df1, df2)

    # filter out couples including unknown nodes
    inference_pairs_df = delete_invalid_couples(inference_pairs_df, nodes_df)

    inference_pairs_df.to_csv(OUTPUT_CSV, index=False)

    print(len(inference_pairs_df), "inference pairs generated.")
    print(f"Inference pairs saved to {OUTPUT_CSV}")
 
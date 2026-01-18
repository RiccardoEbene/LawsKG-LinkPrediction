import torch
import pandas as pd
import dgl
from neo4j import GraphDatabase

def insert_new_links(new_links: str, uri: str, k: int = 5000, wrong: bool = False):
    driver = GraphDatabase.driver(uri, auth=("", ""))

    new_links_df = pd.read_csv(new_links)[['node_1', 'node_2']]
    if not wrong:
        links = new_links_df.head(k)
        rel_type = "RELATED"
    else:
        links = new_links_df.tail(k*10).sample(n=k).reset_index(drop=True)
        rel_type = "WRONG_RELATED"

    print(f"Inserting {len(links)} new links of type {rel_type}...")

    links = links.to_dict('records')
    
    query = """
        UNWIND $links AS link
        MATCH (a1:LawUnit {id: link.node_1}), (a2:LawUnit {id: link.node_2})
        WHERE a1.id <> a2.id AND
            NOT EXISTS((a1)-[]-(a2))
        MERGE (a1)-[:"""+rel_type+"""]->(a2)
        """
    
    with driver.session() as session:
        session.run(query, links=links)
    
    driver.close()


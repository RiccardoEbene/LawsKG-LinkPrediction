import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from scripts.set_new_embeddings import update_embeddings_in_db

def clean_up(updated_nodes: list, driver_uri: str, auth: tuple):
    """Deletes the newly inserted links and restores old embeddings."""
    driver = GraphDatabase.driver(driver_uri, auth=auth)
    
    # Delete newly inserted RELATED and WRONG_RELATED links
    delete_query = """
        MATCH ()-[r:RELATED|WRONG_RELATED]->()
        DELETE r
    """
    
    with driver.session() as session:
        session.run(delete_query)
    
    print("Deleted newly inserted links.")
    
    restore_query = """
        UNWIND $updated_nodes AS node_id
        MATCH (n:LawUnit {id: node_id})
        SET n.new_embedding = n.embedding
    """
    
    with driver.session() as session:
        session.run(restore_query, updated_nodes=updated_nodes)
    
    print("Restored old embeddings.")
    
    driver.close()
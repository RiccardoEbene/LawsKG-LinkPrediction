import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from scripts.set_new_embeddings import update_embeddings_in_db

def clean_up(old_emb: str, driver_uri: str, auth: tuple):
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
    
    # Restore old embeddings
    update_embeddings_in_db(
        npy_path=old_emb,
        driver_uri=driver_uri,
        auth=auth
    )
    
    print("Restored old embeddings.")
    
    driver.close()
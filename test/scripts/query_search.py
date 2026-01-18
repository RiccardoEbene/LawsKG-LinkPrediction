import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
import pandas as pd
import csv

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_embedding(input_text, model_name='intfloat/multilingual-e5-large-instruct', device='cpu'):
    """
    Compute embeddings for an input text using a pre-trained transformer model.

    Args:
        input_text (str): Input text to be embedded.
        model_name (str): Name of the pre-trained transformer model.
        device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
        Tensor: A tensor containing the embedding for the input text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="auto")

    all_embeddings = []


    ## Tokenize input text
    encoded_input = tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    batch_dict = {key: tensor.cuda() for key, tensor in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**batch_dict)
    
    embeddings = average_pool(model_output.last_hidden_state, batch_dict['attention_mask'])
    
    return embeddings.tolist()[0]

def perform_vector_search(query_text, year, output_csv_path, driver_uri, auth):
    # Compute embedding for the query
    embedding = compute_embedding(query_text, device='cuda')
    
    driver = GraphDatabase.driver(driver_uri, auth=auth)

    # Perform vector search
    result, _, _ = driver.execute_query("""
            WITH 500 as k
            CALL vector_search.search('UnitEmbedding', k, $embedding) 
            YIELD node, similarity
            MATCH (l:Law)-[:HAS_ARTICLE]->(node)
            WHERE l.publicationDate > localDateTime({year:$year, month:1, day:1}) 
            RETURN node.id AS id, node.title as title, similarity
            """, embedding=embedding, year=year)

    # Save data to csv
    data = [
        {
            "rank": idx + 1,
            "node_id": record["id"],
            "title": record["title"],
            "similarity": record["similarity"]
        }
        for idx, record in enumerate(result)
    ]

    df = pd.DataFrame(data)

    # csv.QUOTE_NONNUMERIC puts quotes around strings (like titles) but not numbers.
    df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Saved {len(df)} results to {output_csv_path}")

    
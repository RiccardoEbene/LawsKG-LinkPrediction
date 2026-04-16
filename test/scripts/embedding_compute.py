import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
from tqdm import tqdm
import os
import numpy as np
from collections import Counter

LEGAL_STOP_TOPICS = {
    'definizioni', 'sanzioni', 'modifiche legislative', 'normativa eu', 'direttiva ue', 'nan', 'ue',
    'regolamento ue', 'normativa nazionale', 'governo', 'oggetto', 
    'unione europea', 'abrogazioni', 'disposizioni transitorie', 
    'comunicazione', 'controlli', 'obblighi', 'ambito di applicazione',
    'entrata in vigore', 'disposizioni finanziarie', 'copertura finanziaria',
    'norme finali', 'disposizioni generali', 'attuazione', 'decreto legislativo',
    'gazzetta ufficiale', 'pubblicazione', 'comma', 'articolo', 'legge',
    'criteri', 'modalità', 'procedure', 'requisiti', 'termini'
}

def clean_list_topics(topic_list, k=10):
    if isinstance(topic_list, list):
        # Remove invalid entries from the list
        filtered_list = [t for t in topic_list if t != ', ' and t != ''] # and t.lower().strip() not in LEGAL_STOP_TOPICS]

        # Get the top-k most common topics
        topic_counts = Counter(filtered_list)
        top_k_topics = [topic for topic, count in topic_counts.most_common(k)]

        # Converts list of string into single string with values separated by ', '
        text = ', '.join(top_k_topics) 
    elif topic_list is None or topic_list == "":
        # Handles None or empty string case
        text = '' 
    else:
        # If it's already a string
        text = str(topic_list)
    
    # Perform the replacement only if text is not empty
    return text.replace('nessun topic', '')
def create_doc_article(row):
    
    if row.CitTopics is None or row.CitTopics == "":
        return f'''
    Argomenti: {clean_list_topics(row.ArtTopics)}
    Titolo dell'articolo: {row.Title}
    Argomenti della legge: {clean_list_topics(row.TopicLaw)}
    Titolo della legge: {row.LawTitle}'''
    else:
        return f'''
    Argomenti: {clean_list_topics(row.ArtTopics)}
    Titolo dell'articolo: {row.Title}
    Argomenti della legge: {clean_list_topics(row.TopicLaw)}
    Titolo della legge: {row.LawTitle}
    Argomenti articoli citati nella legge: {clean_list_topics(row.CitTopics)}'''
    
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_and_save_embeddings(input_edges_path, n_inserted_links, driver_uri, auth):
    driver = GraphDatabase.driver(driver_uri, auth=auth)

    input_edges = input_edges_path

    df = pd.read_csv(input_edges)

    inserted_links = df[:n_inserted_links]

    nodes_1 = set(inserted_links['node_1'].tolist())
    nodes_2 = set(inserted_links['node_2'].tolist())

    node_ids = list(nodes_1.union(nodes_2))

    # Re-compute embeddings for the new nodes (both node_1 and node_2)
    df = driver.execute_query("""
    MATCH (a:LawUnit)<-[r1]-(l:NationalLaw)
    WHERE a.id in $node_ids
    AND (type(r1) = "HAS_ATTACHMENT" OR type(r1) = "HAS_ARTICLE")
    OPTIONAL MATCH (a)-[r2]->(a2:LawUnit)
    WHERE (type(r2) = 'ABROGATES' OR type(r2) = 'AMENDS' OR type(r2) = "INTRODUCES" or type(r2) = "CITES" or type(r2) = "RELATED")
    WITH a, l, a2.allTopics AS CitTopics
    WITH a, l, COLLECT(CitTopics) AS CitTopics
    RETURN l.title as LawTitle, 
        a.id AS ID, 
        a.title AS Title, a.allTopics AS ArtTopics,
        REDUCE(concatenated = "", topic IN l.mainTopic | concatenated + CASE concatenated WHEN "" THEN "" ELSE ", " END + topic) AS TopicLaw,
        REDUCE(concatenated = "", topic IN CitTopics | concatenated + CASE concatenated WHEN "" THEN "" ELSE ", " END + topic) AS CitTopics
    """, node_ids=node_ids)

    df = pd.DataFrame(df[0], columns=df[-1]).fillna('')


    if len(df) > 0:
        df['text'] = df.apply(create_doc_article, axis = 1)

        # Persist generated texts for inspection/debugging
        # df['text'].to_csv("debug/new_pesticidi_text.txt", index=False, header=False)
        
        input_texts = df.text.to_list()

        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
        model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct').cuda()

        new_embeddings = {}
        
        for j in range(len(input_texts)):
            ##### Tokenize the input texts
            batch_dict = tokenizer(input_texts[j], max_length=512, padding=True, truncation=True, return_tensors='pt')

            # debug
            used_text = tokenizer.decode(batch_dict['input_ids'][0], skip_special_tokens=True)
            print(f"Used text in iter {j}: {used_text}\n\n")

            batch_dict = {key: tensor.cuda() for key, tensor in batch_dict.items()}
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            ##### Normalize embeddings
            # embeddings = F.normalize(embeddings, p=2, dim=1)

            id = df.loc[j,'ID']
            embedding = embeddings.tolist()[0]
                    
            new_embeddings[id] = embedding

        # Save new embeddings to a file
        # np.save(output_npy_path, new_embeddings)

        return node_ids, new_embeddings


def debug_old_embedding(input_edges_path, driver_uri, auth):
    """Check the added topics in the old embedding case"""
    driver = GraphDatabase.driver(driver_uri, auth=auth)

    input_edges = input_edges_path

    df = pd.read_csv(input_edges)

    nodes_1 = set(df['node_1'].tolist())

    # Embedding for new nodes
    df = driver.execute_query("""
    MATCH (a:LawUnit)<-[r1]-(l:NationalLaw)
    WHERE a.id in $node_ids
    AND (type(r1) = "HAS_ATTACHMENT" OR type(r1) = "HAS_ARTICLE")
    OPTIONAL MATCH (a)-[r2]->(a2:LawUnit)
    WHERE (type(r2) = 'ABROGATES' OR type(r2) = 'AMENDS' OR type(r2) = "INTRODUCES" or type(r2) = "CITES")
    WITH a, l, a2.allTopics AS CitTopics
    WITH a, l, COLLECT(CitTopics) AS CitTopics
    RETURN l.title as LawTitle, 
        a.id AS ID, 
        a.title AS Title, a.allTopics AS ArtTopics,
        REDUCE(concatenated = "", topic IN l.mainTopic | concatenated + CASE concatenated WHEN "" THEN "" ELSE ", " END + topic) AS TopicLaw,
        REDUCE(concatenated = "", topic IN CitTopics | concatenated + CASE concatenated WHEN "" THEN "" ELSE ", " END + topic) AS CitTopics
    """, node_ids=list(nodes_1))

    df = pd.DataFrame(df[0], columns=df[-1]).fillna('')


    if len(df) > 0:
        df['text'] = df.apply(create_doc_article, axis = 1)

        # Persist generated texts for inspection/debugging
        # df['text'].to_csv("debug/new_pesticidi_text.txt", index=False, header=False)
        
        input_texts = df.text.to_list()

        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
        model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct').cuda()

        new_embeddings = {}
        
        for j in range(len(input_texts)):

            ##### Tokenize the input texts
            batch_dict = tokenizer(input_texts[j], max_length=512, padding=True, truncation=True, return_tensors='pt')

            # debug
            used_text = tokenizer.decode(batch_dict['input_ids'][0], skip_special_tokens=True)
            print(f"Used text in iter {j}: {used_text}\n\n")

if __name__ == "__main__":
    debug_old_embedding(
        input_edges_path="output/pairs_lavoro_ranked_full.csv", 
        driver_uri="bolt://localhost:23034", 
        auth=("", "")
    )




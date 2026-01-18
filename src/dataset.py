import pandas as pd
import torch
import dgl
import pickle
import os
import numpy as np
from sklearn.preprocessing import normalize

def load_base_graph(nodes_path, edges_path, save_map_path=None):
    print(f"Loading nodes from {nodes_path}...")
    embeddings_df = pd.read_parquet(nodes_path)

    ### BEST PRACTICE: Ensure uniqueness of node_ids to prevent mapping errors
    if embeddings_df['node_id'].duplicated().any():
        print("Warning: Duplicate node IDs found in embedding file. Dropping duplicates.")
        embeddings_df = embeddings_df.drop_duplicates(subset=['node_id'])
    ###

    # Convert embeddings to tensor
    # Stack the list of floats into a 2D array
    embedding_array = np.stack(embeddings_df['embedding'].values)

    print("Normalizing embeddings (L2)...")
    # So that each embedding has unit norm
    embedding_array = normalize(embedding_array, norm='l2', axis=1)

    embedding_tensor = torch.tensor(embedding_array, dtype=torch.float32)
    
    print(f"Loading edges from {edges_path}...")
    edges_df = pd.read_csv(edges_path)

    # Create mapping
    print("Creating node mapping...")
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(embeddings_df['node_id'])}
    
    '''
    # Save mapping for inference later
    if save_map_path:
        with open(save_map_path, 'wb') as f:
            pickle.dump(node_id_to_idx, f)
        print(f"Node mapping saved to {save_map_path}")
    '''

    ### 1. SAFETY CHECK: Filter edges where nodes are missing from the embedding file
    initial_count = len(edges_df)
    valid_src = edges_df['node_1'].isin(node_id_to_idx)
    valid_dst = edges_df['node_2'].isin(node_id_to_idx)
    edges_df = edges_df[valid_src & valid_dst]
    
    if len(edges_df) < initial_count:
        print(f"Dropped {initial_count - len(edges_df)} edges connecting to unknown nodes.")
    ###

    src_indices = edges_df['node_1'].map(node_id_to_idx).values
    dst_indices = edges_df['node_2'].map(node_id_to_idx).values

    g = dgl.graph((src_indices, dst_indices), num_nodes=len(embeddings_df))

    # make bidirectional
    g = dgl.to_bidirected(g)

    # colapse multiple edges
    g = dgl.to_simple(g)

    # add self loops
    g = dgl.add_self_loop(g)

    # Assign embeddings to nodes
    g.ndata["feat"] = embedding_tensor
    
    return g, node_id_to_idx


def load_train_test_split(label_path, node_id_to_idx, g_num_nodes, test_ratio=0.1):
    print(f"Loading labeled data from {label_path}...")
    in_notes_edges_df = pd.read_csv(label_path)

    ### Filter out edges with unknown nodes
    valid_mask = (in_notes_edges_df['node_1'].isin(node_id_to_idx)) & \
                 (in_notes_edges_df['node_2'].isin(node_id_to_idx))
    
    if not valid_mask.all():
        print(f"Dropped {len(in_notes_edges_df) - valid_mask.sum()} pairs with unknown nodes.")
        in_notes_edges_df = in_notes_edges_df[valid_mask]
    ###

    # Shuffle
    in_notes_edges_df = in_notes_edges_df.sample(frac=1, random_state=42).reset_index(drop=True)

    src_indices = torch.tensor(in_notes_edges_df['node_1'].map(node_id_to_idx).values, dtype=torch.int64)
    dst_indices = torch.tensor(in_notes_edges_df['node_2'].map(node_id_to_idx).values, dtype=torch.int64)
    labels = in_notes_edges_df['label'].to_numpy().astype(np.float32)

    n_edges = len(src_indices)
    test_size = int(test_ratio * n_edges)

    train_u, train_v = src_indices[test_size:], dst_indices[test_size:]
    test_u, test_v = src_indices[:test_size], dst_indices[:test_size:]
    
    train_labels = torch.from_numpy(labels[test_size:])
    test_labels = torch.from_numpy(labels[:test_size])

    train_g = dgl.graph((train_u, train_v), num_nodes=g_num_nodes)
    test_g = dgl.graph((test_u, test_v), num_nodes=g_num_nodes)
    
    train_g.edata['label'] = train_labels
    test_g.edata['label'] = test_labels

    return train_g, test_g
import argparse
import torch
import pandas as pd
import dgl
import pickle
from torch.nn.functional import sigmoid
from src.models import GraphSAGE, DotPredictor
from src.dataset import load_base_graph
from src.utils import get_device

def inference(args):
    device = get_device()
    
    '''
    # 1. Load Mapping (Crucial)
    with open(args.map_path, 'rb') as f:
        node_id_to_idx = pickle.load(f)
    '''

    # 2. Load Base Graph Structure (needed for message passing)
    # We don't strictly need to reload the edges DF here if we pickled the graph, 
    # but for simplicity, we rebuild g to ensure consistency with training.
    g, _ = load_base_graph(args.nodes_path, args.edges_path)
    g = g.to(device)
    num_nodes = g.num_nodes()

    # 3. Load Model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    # Extract config from checkpoint if saved, otherwise assume defaults
    input_dim = checkpoint['config']['input_dim']
    hidden_dim = checkpoint['config']['hidden_dim']
    node_id_to_idx = checkpoint['node_map']
    
    model = GraphSAGE(num_nodes, input_dim, hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    pred = DotPredictor().to(device)

    # 4. Load Inference Data
    print(f"Loading inference pairs from {args.input_csv}")
    test_edges_df = pd.read_csv(args.input_csv)
    
    # Map IDs
    # Note: This might fail if test set contains nodes not in training set. 
    # You might want to add error handling here.
    try:
        src_indices = torch.tensor(test_edges_df['node_1'].map(node_id_to_idx).values, dtype=torch.int64).to(device)
        dst_indices = torch.tensor(test_edges_df['node_2'].map(node_id_to_idx).values, dtype=torch.int64).to(device)
    except KeyError:
        print("Error: Inference file contains nodes not found in the training graph.")
        return

    test_g = dgl.graph((src_indices, dst_indices), num_nodes=num_nodes).to(device)

    # 5. Predict
    with torch.no_grad():
        # Compute embeddings on full graph
        h = model(g)
        # Predict on specific edges
        scores = pred(test_g, h)

    # 6. Save Results
    test_edges_df['score'] = scores.cpu().numpy()
    test_edges_df['score_prob'] = sigmoid(scores.detach()).cpu().numpy()
    if 'similarity' in test_edges_df.columns:
        test_edges_df['similarity'] = test_edges_df['similarity'] 
        
    test_edges_df.sort_values(by='score', ascending=False, inplace=True)
    
    test_edges_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")
    print(test_edges_df.head())

def inference2(args):
    device = get_device()
    
    '''
    # 1. Load Mapping (Crucial)
    with open(args.map_path, 'rb') as f:
        node_id_to_idx = pickle.load(f)
    '''

    # 2. Load Base Graph Structure (needed for message passing)
    # We don't strictly need to reload the edges DF here if we pickled the graph, 
    # but for simplicity, we rebuild g to ensure consistency with training.
    g, node_id_to_idx = load_base_graph(args.nodes_path, args.edges_path)
    g = g.to(device)
    num_nodes = g.num_nodes()

    # 3. Load Model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    # Extract config from checkpoint if saved, otherwise assume defaults
    input_dim = checkpoint['config']['input_dim']
    hidden_dim = checkpoint['config']['hidden_dim']
    # node_id_to_idx = checkpoint['node_map']
    
    model = GraphSAGE(num_nodes, input_dim, hidden_dim, pretrained_emb=g.ndata['feat'])
    state_dict = checkpoint['model_state_dict']
    if 'node_emb.weight' in state_dict:
        del state_dict['node_emb.weight']
            
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    pred = DotPredictor().to(device)

    # 4. Load Inference Data
    print(f"Loading inference pairs from {args.input_csv}")
    test_edges_df = pd.read_csv(args.input_csv)
    
    # Map IDs
    # Note: This might fail if test set contains nodes not in training set. 
    # You might want to add error handling here.
    try:
        src_indices = torch.tensor(test_edges_df['node_1'].map(node_id_to_idx).values, dtype=torch.int64).to(device)
        dst_indices = torch.tensor(test_edges_df['node_2'].map(node_id_to_idx).values, dtype=torch.int64).to(device)
    except KeyError:
        print("Error: Inference file contains nodes not found in the training graph.")
        return

    test_g = dgl.graph((src_indices, dst_indices), num_nodes=num_nodes).to(device)

    # 5. Predict
    with torch.no_grad():
        # Compute embeddings on full graph
        h = model(g)
        # Predict on specific edges
        scores = pred(test_g, h)

    # 6. Save Results
    test_edges_df['score'] = scores.cpu().numpy()
    test_edges_df['score_prob'] = sigmoid(scores.detach()).cpu().numpy()
    if 'similarity' in test_edges_df.columns:
        test_edges_df['similarity'] = test_edges_df['similarity'] 
        
    test_edges_df.sort_values(by='score', ascending=False, inplace=True)
    
    test_edges_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")
    print(test_edges_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes_path', type=str, default='data/nodes.parquet')
    parser.add_argument('--edges_path', type=str, default='data/edges.csv')
    parser.add_argument('--input_csv', type=str, default='data/inference_pairs_nucleare.csv')
    parser.add_argument('--output_csv', type=str, default='output/NEW_pairs_nucleare_ranked.csv')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_emb_tuned.pth')
    # parser.add_argument('--map_path', type=str, default='output/node_map.pkl')
    
    args = parser.parse_args()
    inference(args)
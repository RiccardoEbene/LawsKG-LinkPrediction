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
    """
    Perform a forward pass through the model to compute node embeddings, then save them on db.
    
    :param args: Argument parser object with necessary attributes.
    """
    device = get_device()

    model_path = args.model_path
    
    # Load Whole Graph for Embedding Computation
    g, node_id_to_idx = load_base_graph(args.nodes_path, args.edges_path)
    num_nodes = g.num_nodes() # this is the number of nodes in the full graph (isolated nodes included)

    # Load Inference Data
    print(f"Loading inference pairs from {args.input_csv}")
    test_edges_df = pd.read_csv(args.input_csv)
    
    # Map IDs
    # This might fail if test set contains nodes not in training set.
    try:
        src_indices = torch.tensor(test_edges_df['node_1'].map(node_id_to_idx).values, dtype=torch.int64)
        dst_indices = torch.tensor(test_edges_df['node_2'].map(node_id_to_idx).values, dtype=torch.int64)
    except KeyError:
        print("Error: Inference file contains nodes not found in the mapping.")
        return

    test_g = dgl.graph((src_indices, dst_indices), num_nodes=num_nodes)

    # Load Model
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint['config']['input_dim']
    hidden_dim = checkpoint['config']['hidden_dim']
    
    # We will pass features dynamically during the loop. Hence, no pretrained embeddings here.
    model = GraphSAGE(num_nodes, input_dim, hidden_dim, pretrained_emb=None)
    pred = DotPredictor()
    
    state_dict = checkpoint['model_state_dict']
    if 'node_emb.weight' in state_dict:
        del state_dict['node_emb.weight']
            
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Identify number of layers from the model to configure the sampler correctly
    n_layers = 2 
    
    # FullNeighborSampler ensures we aggregate from ALL neighbors, 
    # replicating the exact math of a full-graph forward pass.
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    
    # Create DataLoader for all nodes
    dataloader = dgl.dataloading.DataLoader(
        g, 
        torch.arange(num_nodes), # We want embeddings for ALL nodes
        sampler,
        batch_size=2048, # (1024 - 4096 usually safe)
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    # Prepare a tensor on CPU to store the results
    all_embeddings = torch.zeros(num_nodes, hidden_dim)
    
    print("Starting mini-batch inference...")
    with dataloader.enable_cpu_affinity():
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in dataloader:
                # Move only the necessary small blocks to GPU
                blocks = [b.to(device) for b in blocks]
                
                # Extract features for the input nodes of the first layer
                # These are copied from RAM to VRAM just for this batch
                input_features = blocks[0].srcdata['feat']

                # Forward pass
                batch_emb = model(blocks, input_features)
                
                # Save results back to CPU
                all_embeddings[output_nodes] = batch_emb.cpu()

            # Predict on test edges
            scores = pred(test_g, all_embeddings)
            
    # Save Predictions
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
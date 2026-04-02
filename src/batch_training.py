import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import itertools
import dgl
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.models import GraphSAGE, DotPredictor, NodeMLP
from src.dataset import load_base_graph, load_train_test_split
from src.utils import compute_loss, compute_auc, compute_recall, compute_f1, save_checkpoint, get_device, set_seed

USE_PRETRAINED = True

def main(args):
    # Set seeds early so data loading / model init are deterministic
    set_seed(args.seed)

    writer = SummaryWriter(log_dir=args.log_dir)

    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Data
    g, node_map = load_base_graph(args.nodes_path, args.edges_path)
    train_g, test_g = load_train_test_split(args.label_path, node_map, g.num_nodes())

    # Keep base graph (g) on CPU to save memory. 
    # Mini-batching will automatically move only the necessary sampled chunks to the GPU.
    if USE_PRETRAINED:
        node_features = g.ndata['feat'] # Do not use .to(device) here
    else:
        node_features = None

    # 2. Calculate Weights (Using train_g)
    num_pos = torch.sum(train_g.edata['label'] == 1)
    num_neg = torch.sum(train_g.edata['label'] == 0)
    pos_weight = torch.tensor(num_neg / (num_pos + 1e-8)).to(device)
    print(f"Pos weight: {pos_weight.item():.2f}")

    # 3. Setup PyTorch DataLoaders for Edges
    # Extract edges and labels directly to feed into standard PyTorch TensorDatasets
    train_u, train_v = train_g.edges()
    train_labels = train_g.edata['label']
    train_dataset = TensorDataset(train_u, train_v, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_u, test_v = test_g.edges()
    test_labels = test_g.edata['label']
    test_dataset = TensorDataset(test_u, test_v, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Model
    if args.ablation:
        print("Running Ablation (NodeMLP)...")
        # Note: Ensure NodeMLP is also updated to accept blocks if you run ablation in mini-batch mode
        model = NodeMLP(
                    num_nodes=g.num_nodes(), 
                    in_feats=args.input_dim, 
                    h_feats=args.hidden_dim,
                    pretrained_emb=node_features
                ).to(device)
    else:
        print("Running GraphSAGE...")
        model = GraphSAGE(
                    num_nodes=g.num_nodes(), 
                    in_feats=args.input_dim, 
                    h_feats=args.hidden_dim, 
                    pretrained_emb=node_features,
                    aggregator_type=args.aggregator
                ).to(device)
        
    # We no longer need the DotPredictor module since we are doing manual tensor dot products
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    warmup_epochs = 2
    scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
    
    # 5. Setup DGL Sampler
    sampler = dgl.dataloading.NeighborSampler([10, 10])

    # 6. Training Loop
    best_f1 = 0
    
    for e in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_u, batch_v, batch_labels in train_dataloader:
            batch_u = batch_u.to(device)
            batch_v = batch_v.to(device)
            batch_labels = batch_labels.to(device)
            
            # Step A: Identify unique nodes needed for this batch of edges
            seed_nodes = torch.unique(torch.cat([batch_u, batch_v]))
            
            # Step B: Sample computational blocks ONLY from the encoder graph (g)
            input_nodes, output_nodes, blocks = sampler.sample_blocks(g, seed_nodes.cpu())
            blocks = [b.to(device) for b in blocks]
            
            # Step C: Forward pass through GraphSAGE
            input_features = blocks[0].srcdata['feat'] if USE_PRETRAINED else None
            batch_emb = model(blocks, input_features) 
            
            # Step D: Map global node IDs to their local indices in the output embedding
            rev_index = torch.empty(g.num_nodes(), dtype=torch.long, device=device)
            rev_index[seed_nodes] = torch.arange(len(seed_nodes), device=device)
            
            local_u = rev_index[batch_u]
            local_v = rev_index[batch_v]
            
            emb_u = batch_emb[local_u]
            emb_v = batch_emb[local_v]
            
            # Step E: Predictor (Dot Product on the mapped embeddings)
            scores = (emb_u * emb_v).sum(dim=1)
            
            # Step F: Loss and Backward
            loss = compute_loss(scores, batch_labels, pos_weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # End of Epoch
        scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        
        # Log Training Loss to TensorBoard 
        writer.add_scalar('Train/Loss', avg_loss, e)
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], e)

        # 7. Evaluation Loop
        if e % args.eval_every == 0:
            model.eval()
            all_test_scores = []
            all_test_labels = []
            
            with torch.no_grad():
                for batch_u, batch_v, batch_labels in test_dataloader:
                    batch_u = batch_u.to(device)
                    batch_v = batch_v.to(device)
                    
                    seed_nodes = torch.unique(torch.cat([batch_u, batch_v]))
                    input_nodes, output_nodes, blocks = sampler.sample_blocks(g, seed_nodes.cpu())
                    blocks = [b.to(device) for b in blocks]
                    
                    input_features = blocks[0].srcdata['feat'] if USE_PRETRAINED else None
                    batch_emb = model(blocks, input_features)
                    
                    rev_index = torch.empty(g.num_nodes(), dtype=torch.long, device=device)
                    rev_index[seed_nodes] = torch.arange(len(seed_nodes), device=device)
                    
                    local_u = rev_index[batch_u]
                    local_v = rev_index[batch_v]
                    
                    emb_u = batch_emb[local_u]
                    emb_v = batch_emb[local_v]
                    
                    scores = (emb_u * emb_v).sum(dim=1)
                    
                    all_test_scores.append(scores.cpu())
                    all_test_labels.append(batch_labels)
            
            test_scores = torch.cat(all_test_scores, dim=0)
            test_labels = torch.cat(all_test_labels, dim=0)
            
            f1, best_thresh = compute_f1(test_scores, test_labels)
            auc = compute_auc(test_scores, test_labels)
            
            print(f"Epoch {e} | Loss: {avg_loss:.4f} | Test F1: {f1:.4f} | Test AUC: {auc:.4f} | Best Thresh: {best_thresh:.4f}")
            
            # Log Evaluation Metrics to TensorBoard
            writer.add_scalar('Test/F1', f1, e)
            writer.add_scalar('Test/AUC', auc, e)
            writer.add_scalar('Test/Best_Threshold', best_thresh, e)

            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(model, optimizer, vars(args), node_map, args.model_save_path)
                print(f"  -> Model saved to {args.model_save_path}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--nodes_path', type=str, default='data/train_nodes.parquet')
    parser.add_argument('--edges_path', type=str, default='data/edges.csv')
    parser.add_argument('--label_path', type=str, default='data/in_notes_set_labeled.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints/graphsage_batch.pth')
    # parser.add_argument('--map_save_path', type=str, default='checkpoints/node_map.pkl')
    
    # Hyperparams
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--aggregator', type=str, default='mean')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for mini-batch training')
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='logs/all_edges_batch_run')
    parser.add_argument('--ablation', action='store_true', help="Use NodeMLP instead of GraphSAGE")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)
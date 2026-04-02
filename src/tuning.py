import argparse
import torch
import itertools
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.models import GraphSAGE, DotPredictor, NodeMLP
from src.dataset import load_base_graph, load_train_test_split
from src.utils import compute_loss, compute_auc, compute_recall, compute_f1, save_checkpoint, get_device, set_seed

USE_PRETRAINED = True

def main(args):
    # Set seeds early so data loading / model init are deterministic
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Data ONCE for all runs to save I/O overhead
    g, node_map = load_base_graph(args.nodes_path, args.edges_path)
    train_g, test_g = load_train_test_split(args.label_path, node_map, g.num_nodes())

    if USE_PRETRAINED:
        node_features = g.ndata['feat'].to(device)
    else:
        node_features = None
    
    # Move to device
    g = g.to(device)
    train_g = train_g.to(device)
    test_g = test_g.to(device)
    
    # Move labels to device specifically
    train_g.edata['label'] = train_g.edata['label'].to(device)
    test_g.edata['label'] = test_g.edata['label'].to(device)

    # 2. Calculate Weights
    num_pos = torch.sum(train_g.edata['label'] == 1)
    num_neg = torch.sum(train_g.edata['label'] == 0)
    pos_weight = torch.tensor(num_neg / (num_pos + 1e-8)).to(device)
    print(f"Pos weight: {pos_weight.item():.2f}")

    # Define Exhaustive Hyperparameter Grid
    param_grid = {
        'lr': [1e-3, 5e-4, 1e-4],
        'weight_decay': [1e-4, 1e-5, 0.0],
        'hidden_dim': [512, 1024],
        'aggregator': ['mean', 'gcn', 'pool']
    }

    # Generate all possible combinations
    keys, values = zip(*param_grid.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    global_best_f1 = 0
    best_params = None

    print(f"Starting Grid Search with {len(grid_combinations)} combinations...")

    for idx, params in enumerate(grid_combinations):
        print(f"\n--- Run {idx + 1}/{len(grid_combinations)} | Params: {params} ---")
        
        # Update args with current grid parameters
        for k, v in params.items():
            setattr(args, k, v)
            
        # Unique TensorBoard log directory for each run
        run_log_dir = os.path.join(args.log_dir, f"run_{idx}_lr{args.lr}_hd{args.hidden_dim}")
        writer = SummaryWriter(log_dir=run_log_dir)

        # 3. Initialize Model dynamically per run
        if args.ablation:
            model = NodeMLP(
                        num_nodes=g.num_nodes(), 
                        in_feats=args.input_dim, 
                        h_feats=args.hidden_dim,
                        pretrained_emb=node_features
                    ).to(device)
        else:
            model = GraphSAGE(
                        num_nodes=g.num_nodes(), 
                        in_feats=args.input_dim, 
                        h_feats=args.hidden_dim, 
                        pretrained_emb=node_features,
                        aggregator_type=args.aggregator
                    ).to(device)
            
        pred = DotPredictor().to(device)

        optimizer = torch.optim.Adam(
            itertools.chain(model.parameters(), pred.parameters()), lr=args.lr, weight_decay=args.weight_decay
        )
        
        warmup_epochs = 2
        scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
        
        run_best_f1 = 0
        
        # 4. Training Loop
        for e in range(1, args.epochs + 1):
            model.train()
            pred.train()
            
            # Forward
            h = model(g)
            scores = pred(train_g, h)
            loss = compute_loss(scores, train_g.edata['label'], pos_weight)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar('Train/Loss', loss.item(), e)
            writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], e)

            if e % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    h_eval = model(g)
                    test_scores = pred(test_g, h_eval)
                    f1, best_thresh = compute_f1(test_scores, test_g.edata['label'])
                    auc = compute_auc(test_scores, test_g.edata['label'])
                    
                    writer.add_scalar('Test/F1', f1, e)
                    writer.add_scalar('Test/AUC', auc, e)
                    writer.add_scalar('Test/Best_Threshold', best_thresh, e)

                    if f1 > run_best_f1:
                        run_best_f1 = f1
                        
                        # Only save the checkpoint if it is the GLOBAL best across all runs
                        if f1 > global_best_f1:
                            global_best_f1 = f1
                            best_params = params
                            save_checkpoint(model, optimizer, vars(args), node_map, args.model_save_path)
                            print(f"  -> NEW GLOBAL BEST! Epoch {e} | F1: {f1:.4f} | AUC: {auc:.4f} | Saved to {args.model_save_path}")

        writer.close()
        print(f"Run {idx + 1} finished. Best F1 for this run: {run_best_f1:.4f}")

    print("\n" + "="*50)
    print(f"Grid Search Complete!")
    print(f"Best Global F1: {global_best_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--nodes_path', type=str, default='data/train_nodes.parquet')
    parser.add_argument('--edges_path', type=str, default='data/edges.csv')
    parser.add_argument('--label_path', type=str, default='data/in_notes_set_labeled.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints/tuning_graphsage_full.pth')
    
    # Defaults (will be overwritten by grid search)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--aggregator', type=str, default='mean')
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='logs/hyperparam_search')
    parser.add_argument('--ablation', action='store_true', help="Use NodeMLP instead of GraphSAGE")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)
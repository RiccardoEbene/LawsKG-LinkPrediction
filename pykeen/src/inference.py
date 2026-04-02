import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pykeen.triples import TriplesFactory

def main(args):
    model_dir = Path(args.model_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load Saved Model and TriplesFactory ---
    print(f"Loading model and TriplesFactory from {model_dir}...")
    model_path = model_dir / "trained_model.pkl"
    tf_dir = model_dir / "training_triples"
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    tf = TriplesFactory.from_path_binary(tf_dir)
    
    entity_to_id = tf.entity_to_id
    relation_id = tf.relation_to_id['IN_NOTES'] 
    print(f"Loaded successfully. Relation ID for 'IN_NOTES': {relation_id}")
    
    # --- 2. Load Inference Data ---
    print(f"\nLoading inference data from {args.inference_csv}...")
    df_inference = pd.read_csv(args.inference_csv)
    valid_nodes = set(entity_to_id.keys())
    
    df_inference = df_inference[
        df_inference['node_1'].astype(str).isin(valid_nodes) & 
        df_inference['node_2'].astype(str).isin(valid_nodes)
    ]
    print(f"Valid rows for inference: {len(df_inference)}")

    # --- 3. Prepare Tensors ---
    head_ids = [entity_to_id[str(x)] for x in df_inference['node_1'].values]
    tail_ids = [entity_to_id[str(x)] for x in df_inference['node_2'].values]

    h_tensor = torch.tensor(head_ids, dtype=torch.long, device=device)
    t_tensor = torch.tensor(tail_ids, dtype=torch.long, device=device)
    r_tensor = torch.full_like(h_tensor, relation_id) 

    batch = torch.stack([h_tensor, r_tensor, t_tensor], dim=1)

    # --- 4. Compute Scores ---
    print("Calculating scores...")
    model.eval()
    with torch.no_grad():
        scores = model.score_hrt(batch).squeeze().cpu()

    df_inference["score"] = torch.sigmoid(scores).numpy()
    df_inference = df_inference.sort_values(by="score", ascending=False)
    
    print(f"\n--- Top {args.top_k} Predicted Links ---")
    print(df_inference.head(args.top_k))
    
    df_inference.to_csv(args.out_csv, index=False)
    print(f"\nFull inference results saved to {args.out_csv}")

    # --- 5. Plot Distribution ---
    print("Generating score distribution plot...")
    plt.figure(figsize=(10, 6))
    plt.hist(df_inference['score'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plot_filename = "pykeen/output/inference_results/score_distribution.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using the trained RotatE model.")
    parser.add_argument("--inference_csv", type=str, default="data/inference_pairs_combustibili.csv", help="CSV containing pairs to infer.")
    parser.add_argument("--model_dir", type=str, default="pykeen/output/saved_models/rotate_model", help="Directory where the trained model is saved.")
    parser.add_argument("--out_csv", type=str, default="pykeen/output/inference_results/inference_results_combustibili.csv", help="Output CSV for inference results.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top predictions to print to console.")
    
    args = parser.parse_args()
    main(args)
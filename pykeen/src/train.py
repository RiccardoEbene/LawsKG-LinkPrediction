import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.metrics import roc_auc_score, precision_recall_curve, recall_score

def compute_auc(labels, scores):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.0

def compute_f1_and_threshold(labels, scores):
    scores_norm = torch.sigmoid(scores)
    precisions, recalls, thresholds = precision_recall_curve(labels, scores_norm)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx]
    
    return best_f1, best_thresh

def compute_recall(labels, scores, threshold=0.5):
    probs = torch.sigmoid(scores)
    preds = (probs >= threshold).long()
    return recall_score(labels, preds)

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Data
    print(f"Loading training data from {args.train_csv}...")
    df_train = pd.read_csv(args.train_csv, engine='python')
    triples = df_train[["node_1", "REL", "node_2"]].to_numpy()
    
    train_tf = TriplesFactory.from_labeled_triples(triples)
    test_tf = TriplesFactory.from_labeled_triples(triples[:10]) # dummy test set for pipeline (we will evaluate on custom test set later)

    # Training 
    print(f"Training RotatE model (Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr})...")
    result = pipeline(
        training=train_tf,
        testing=test_tf, # placeholder
        model="RotatE",
        model_kwargs=dict(embedding_dim=args.embedding_dim),
        training_kwargs=dict(num_epochs=args.epochs, batch_size=args.batch_size),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=args.lr),
        negative_sampler="bernoulli",
        random_seed=42,
        device=device,
    )

    # --- 3. Save Model ---
    print(f"Saving model to {out_dir.resolve()}...")
    result.save_to_directory(out_dir)

    # --- 4. Evaluate on Test Set ---
    print(f"\nEvaluating custom metrics on {args.test_csv}...")
    model = result.model
    model.eval()
    
    entity_to_id = train_tf.entity_to_id
    relation_id = train_tf.relation_to_id['IN_NOTES']

    df_test = pd.read_csv(args.test_csv)
    valid_nodes = set(entity_to_id.keys())
    
    df_test = df_test[
        df_test['node_1'].astype(str).isin(valid_nodes) & 
        df_test['node_2'].astype(str).isin(valid_nodes)
    ]
    print(f"Valid rows for testing: {len(df_test)}")

    head_ids = [entity_to_id[str(x)] for x in df_test['node_1'].values]
    tail_ids = [entity_to_id[str(x)] for x in df_test['node_2'].values]

    h_tensor = torch.tensor(head_ids, dtype=torch.long, device=device)
    t_tensor = torch.tensor(tail_ids, dtype=torch.long, device=device)
    r_tensor = torch.full_like(h_tensor, relation_id)

    batch = torch.stack([h_tensor, r_tensor, t_tensor], dim=1)

    with torch.no_grad():
        scores = model.score_hrt(batch).squeeze().cpu()

    labels = df_test['label'].values
    auc = compute_auc(labels, scores)
    best_f1, best_thresh = compute_f1_and_threshold(labels, scores)
    recall = compute_recall(labels, scores, threshold=best_thresh) 

    print("\n--- Evaluation Results ---")
    print(f"AUC:    {auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Best F1:{best_f1:.4f} (at threshold {best_thresh:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RotatE model using PyKEEN.")
    parser.add_argument("--train_csv", type=str, default="pykeen/data/trainingtriples_pykeen.csv", help="Path to training CSV.")
    parser.add_argument("--test_csv", type=str, default="pykeen/data/testingtriples_in_notes.csv", help="Path to testing CSV for evaluation.")
    parser.add_argument("--out_dir", type=str, default="pykeen/output/rotate_model", help="Directory to save the trained model.")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension for RotatE.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    
    args = parser.parse_args()
    main(args)
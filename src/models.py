import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import SAGEConv
import math

class GraphSAGE(nn.Module):
    def __init__(self, num_nodes, in_feats, h_feats, pretrained_emb=None, aggregator_type='mean', dropout=0.5):
        super(GraphSAGE, self).__init__()

        self.pretrained = pretrained_emb is not None

        if self.pretrained:
            self.node_emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
        else:
            # Internal embedding layer
            self.node_emb = nn.Embedding(num_nodes, in_feats)
            nn.init.xavier_uniform_(self.node_emb.weight)
            # bound = 1 / math.sqrt(in_feats)
            # nn.init.uniform_(self.node_emb.weight, -bound, bound)

            self.node_emb.weight.requires_grad = False

        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type=aggregator_type)
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type=aggregator_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x=None):
        """
        :param g: DGLGraph (full batch) OR list of Blocks (mini-batch)
        :param x: Feature tensor (optional for full batch, required for mini-batch)
        """
        
        # Case 1: Mini-batch Inference (g is a list of blocks)
        if isinstance(g, list):
            # If no features are explicitly passed, look up the learned embeddings using the block's original node IDs
            if x is None:
                src_node_ids = g[0].srcdata[dgl.NID]
                h = self.node_emb(src_node_ids)
            else:
                h = x
            
            # Layer 1: Apply to first block
            h = self.conv1(g[0], h)
            h = F.leaky_relu(h, negative_slope=0.2)
            h = self.dropout(h)
            
            # Layer 2: Apply to second block
            h = self.conv2(g[1], h)
            return h

        # Case 2: Full Graph (Training behavior)
        else:
            # If x is not provided, use the internal node_emb
            if x is None:
                x = self.node_emb(g.nodes())

            h = self.conv1(g, x)
            h = F.leaky_relu(h, negative_slope=0.2)
            h = self.dropout(h)
            h = self.conv2(g, h)
            return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]

class NodeMLP(nn.Module):
    """For Ablation Study"""
    def __init__(self, num_nodes, in_feats, h_feats, pretrained_emb=None, dropout=0.5):
        super(NodeMLP, self).__init__()
        
        if pretrained_emb is not None:
            self.node_emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
        else:
            # Random weights (Xavier)
            self.node_emb = nn.Embedding(num_nodes, in_feats)
            nn.init.xavier_uniform_(self.node_emb.weight)
            self.node_emb.weight.requires_grad = False
        
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(h_feats, h_feats)

    def forward(self, g):
            # --- Recupero Feature ---
            node_ids = g.nodes()
            h = self.node_emb(node_ids)

            h = self.linear1(h)
            
            h = F.leaky_relu(h, negative_slope=0.2) 
            
            h = self.dropout(h)
            
            h = self.linear2(h)
            return h
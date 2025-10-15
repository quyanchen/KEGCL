import argparse
import os
import os.path as osp
import random
import yaml
from yaml import SafeLoader
import numpy as np
import scipy
import torch
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.utils import dropout_adj, degree, to_undirected, get_laplacian
import torch.nn.functional as F
import networkx as nx
from scipy.sparse.linalg import eigs, eigsh

from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from model import GConv, Encoder, KEGCL
from functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
# from dataset import get_dataset
from utils import normalize_adjacency_matrix, create_adjacency_matrix, load_adj_neg, Rank


class EarlyStopping:
    
    def __init__(self, patience=50, min_delta=1e-6, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, loss, epoch):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_epoch = epoch
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose and self.counter % 10 == 0:
                print(f'    EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'    Early stopping triggered at epoch {epoch}')
                    print(f'    Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}')
        else:
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


def load_removed_edges_simple(original_ppi_file, subnet_edge_file, node_file):
    
    with open(node_file, 'r') as f:
        subnet_nodes_list = [line.strip() for line in f if line.strip()]
    subnet_nodes_set = set(subnet_nodes_list)
    node2idx = {node: idx for idx, node in enumerate(subnet_nodes_list)}
    
    subnet_edges = set()
    with open(subnet_edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) >= 2:
                a, b = parts[0].strip(), parts[1].strip()
                subnet_edges.add((min(a, b), max(a, b)))

    original_edges = set()
    with open(original_ppi_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) >= 2:
                a, b = parts[0].strip(), parts[1].strip()
                if a in subnet_nodes_set and b in subnet_nodes_set:
                    edge = (min(a, b), max(a, b))
                    original_edges.add(edge)

    removed_edges = original_edges - subnet_edges
    
    return removed_edges, subnet_nodes_set, node2idx


def add_random_removed_edges(edge_index, removed_edges_protein, subnet_nodes, 
                             node2idx, add_ratio=0.1):
    if len(removed_edges_protein) == 0:
        return edge_index
    
    num_to_add = max(1, int(len(removed_edges_protein) * add_ratio))
    
    edges_to_add = random.sample(list(removed_edges_protein), 
                                 min(num_to_add, len(removed_edges_protein)))
    
    new_edges = []
    for a, b in edges_to_add:
        if a in node2idx and b in node2idx:
            idx_a, idx_b = node2idx[a], node2idx[b]
            new_edges.append([idx_a, idx_b])
            new_edges.append([idx_b, idx_a])
    
    if len(new_edges) == 0:
        return edge_index
    
    new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
    augmented_edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)
    
    return augmented_edge_index


def train(model, optimizer, data, drop_edge_rate_1, drop_edge_rate_2, 
         drop_feature_rate_1, drop_feature_rate_2):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0] 
    x_1 = drop_feature(data.x, drop_feature_rate_1)
    x_2 = drop_feature(data.x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1, [2, 2])
    z2 = model(x_2, edge_index_2, [4, 4])

    loss = model.loss(z1, z2, batch_size=None)
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='bio')
    parser.add_argument('--config', type=str, default='param.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--add_edge_ratio', type=float, default=0.1)
    parser.add_argument('--original_ppi', type=str, default='dataset/biogrid.txt')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--min_delta', type=float, default=1e-6)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = config['activation']
    base_model = config['base_model']
    num_layers = config['num_layers']
    dataset = args.dataset
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    drop_scheme = config['drop_scheme']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    rand_layers = config['rand_layers']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)

    for ii in range(255):
        attr_file = f"bio/Attribute_biosl{ii}.txt"
        network_file = f"bio/Network_biosl{ii}.txt"
        node_file = f"bio/bio{ii}_node.txt"
        subnet_edge_file = f"bio/biogrid_{ii}.txt"

        if not os.path.exists(attr_file) or not os.path.exists(network_file):
            continue


        A = np.loadtxt(attr_file, delimiter=" ")
        if A.ndim == 2 and A.shape[1] > 0:
            A = A[:, 1:] if A.shape[1] > 1 else A
        
        N = np.loadtxt(network_file, delimiter=" ")
        edge_index = torch.tensor(np.array(np.nonzero(N)), dtype=torch.long)
        
        removed_edges_protein = set()
        node2idx = {}
        subnet_nodes = set()
        
        if os.path.exists(node_file) and os.path.exists(subnet_edge_file):
            removed_edges_protein, subnet_nodes, node2idx = load_removed_edges_simple(
                original_ppi_file="dataset/biogrid.txt",
                subnet_edge_file=subnet_edge_file,
                node_file=node_file
            )
            print(f"{len(removed_edges_protein)}")
        
        if len(removed_edges_protein) > 0 and args.add_edge_ratio > 0:
            edge_index = add_random_removed_edges(
                edge_index, removed_edges_protein, subnet_nodes, 
                node2idx, add_ratio=args.add_edge_ratio
            )
            print(f"{edge_index.shape[1] // 2}")
        
        from torch_geometric.data import Data
        
        x = torch.tensor(A, dtype=torch.float)
        data = Data(edge_index=edge_index, x=x)
        data = data.to(device)

        encoder = Encoder(data.num_features, num_hidden, get_activation(activation),
                         base_model=GConv, k=num_layers).to(device)
        model = KEGCL(encoder, N, num_hidden, num_proj_hidden, tau).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        log = args.verbose.split(',')

        early_stopping = None
        if args.early_stopping:
            early_stopping = EarlyStopping(
                patience=args.patience, 
                min_delta=args.min_delta,
                verbose='train' in log
            )

        stopped_early = False
        final_epoch = num_epochs
        
        for epoch in range(1, num_epochs + 1):
            loss = train(model, optimizer, data, drop_edge_rate_1, drop_edge_rate_2,
                        drop_feature_rate_1, drop_feature_rate_2)
            
            if 'train' in log:
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
            
            if early_stopping is not None:
                if early_stopping(loss, epoch):
                    stopped_early = True
                    final_epoch = epoch
                    break
            
            if epoch % 1000 == 0:
                z = model(data.x, data.edge_index, [2, 2], final=True).detach().cpu().numpy()
                np.savetxt(f"bio{ii}_{epoch}_embed.txt", z, fmt='%.15f')
        
        print(f"{final_epoch}/{num_epochs}")
        
        model.eval()
        with torch.no_grad():
            z_final = model(data.x, data.edge_index, [2, 2], final=True).detach().cpu().numpy()
        
        final_embed_file = f"bio{ii}_final_embed.txt"
        np.savetxt(final_embed_file, z_final, fmt='%.15f')
        print(f"  - final embedding: {final_embed_file}")
        print(f"{'='*50}\n")

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


def load_removed_edges_and_nodes(original_ppi_file, subnet_edge_file, node_file, go_attr_file):
    
    # 1. Load GO annotations for all proteins
    protein_go = {}
    with open(go_attr_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            protein = parts[0].strip()
            terms_field = parts[1]
            go_terms = terms_field.strip().split()
            
            go_list = []
            for term in go_terms:
                term = term.strip()
                if term.startswith("GO:"):
                    go_list.append(term)
            protein_go[protein] = go_list
    
    
    # 2. Load current subnet nodes
    with open(node_file, 'r') as f:
        subnet_nodes_list = [line.strip() for line in f if line.strip()]
    subnet_nodes_set = set(subnet_nodes_list)
    node2idx = {node: idx for idx, node in enumerate(subnet_nodes_list)}
    
    
    # 3. Load edges retained in the subnet
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
    
    # 4. Load all edges from original PPI
    all_original_edges = set()
    all_nodes_in_ppi = set()
    with open(original_ppi_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) >= 2:
                a, b = parts[0].strip(), parts[1].strip()
                all_original_edges.add((min(a, b), max(a, b)))
                all_nodes_in_ppi.add(a)
                all_nodes_in_ppi.add(b)

    
    # 5. Find removed edges (edges with at least one endpoint in current subnet)
    removed_edges = set()
    for edge in all_original_edges:
        a, b = edge
        # If at least one node is in current subnet and edge is not retained
        if (a in subnet_nodes_set or b in subnet_nodes_set) and edge not in subnet_edges:
            removed_edges.add(edge)
    
    print(f"  [Load] Removed edges: {len(removed_edges)}")
    
    # 6. Find candidate nodes to restore (nodes connected to current subnet via removed edges)
    candidate_removed_nodes = set()
    for edge in removed_edges:
        a, b = edge
        if a not in subnet_nodes_set:
            candidate_removed_nodes.add(a)
        if b not in subnet_nodes_set:
            candidate_removed_nodes.add(b)
    
    print(f"  [Load] Candidate nodes for restoration: {len(candidate_removed_nodes)}")
    
    if len(removed_edges) > 0:
        sample = list(removed_edges)[:3]
        print(f"  [Load] Sample removed edges: {sample}")
    
    return removed_edges, candidate_removed_nodes, subnet_nodes_list, node2idx, protein_go


def restore_edges_and_nodes(edge_index, x, removed_edges, removed_nodes, 
                           subnet_nodes_list, node2idx, protein_go, 
                           attr_go_terms, restore_ratio=0.1):

    if len(removed_edges) == 0:
        return edge_index, x, node2idx, subnet_nodes_list, 0, 0
    
    # 1. Randomly select edges to restore
    num_to_restore = max(1, int(len(removed_edges) * restore_ratio))
    edges_to_restore = random.sample(list(removed_edges), 
                                     min(num_to_restore, len(removed_edges)))
    
    
    # 2. Identify new nodes needed
    new_nodes_needed = set()
    for a, b in edges_to_restore:
        if a not in node2idx:
            new_nodes_needed.add(a)
        if b not in node2idx:
            new_nodes_needed.add(b)
    
    # 3. Create feature vectors for new nodes
    new_node2idx = node2idx.copy()
    new_nodes_list = subnet_nodes_list.copy()
    current_idx = len(subnet_nodes_list)
    new_features = []
    
    # CRITICAL: Use the same GO terms as the original attribute matrix
    num_features = x.shape[1]  # Get feature dimension from existing matrix
    
    for node in sorted(new_nodes_needed):
        new_node2idx[node] = current_idx
        new_nodes_list.append(node)
        
        # Create GO feature vector for this node
        go_terms = protein_go.get(node, [])
        feature_vec = np.zeros(num_features)
        
        for idx, go_term in enumerate(attr_go_terms):
            if go_term in go_terms:
                feature_vec[idx] = 1
        
        new_features.append(feature_vec)
        current_idx += 1
    
    print(f"  [Restore] New nodes to add: {len(new_nodes_needed)}")
    
    # 4. Expand feature matrix
    if len(new_features) > 0:
        new_features_array = np.array(new_features, dtype=np.float32)
        new_features_tensor = torch.tensor(new_features_array, dtype=torch.float)
        new_x = torch.cat([x, new_features_tensor], dim=0)
    else:
        new_x = x
    
    # 5. Add restored edges
    new_edges = []
    successfully_restored = 0
    for a, b in edges_to_restore:
        if a in new_node2idx and b in new_node2idx:
            idx_a = new_node2idx[a]
            idx_b = new_node2idx[b]
            new_edges.append([idx_a, idx_b])
            new_edges.append([idx_b, idx_a])
            successfully_restored += 1
    
    # 6. Merge edges
    if len(new_edges) > 0:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        new_edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)
    else:
        new_edge_index = edge_index
    
    print(f"  [Restore] New graph: nodes={new_x.shape[0]}, edges={new_edge_index.shape[1]//2}")
    
    return new_edge_index, new_x, new_node2idx, new_nodes_list, successfully_restored, len(new_nodes_needed)


def save_restored_subnet(subnet_idx, nodes_list, edge_index, embeddings, output_dir="restored_subnets"):

    os.makedirs(output_dir, exist_ok=True)
    
    # Save node list
    node_file = os.path.join(output_dir, f"biosl{subnet_idx}_restored_nodes.txt")
    with open(node_file, 'w') as f:
        for node in nodes_list:
            f.write(f"{node}\n")
    
    # Save edges (undirected, save each edge once)
    edge_file = os.path.join(output_dir, f"biosl{subnet_idx}_restored_edges.txt")
    edge_set = set()
    edge_list = edge_index.cpu().numpy()
    for i in range(edge_list.shape[1]):
        u, v = edge_list[0, i], edge_list[1, i]
        if u < v:  # Save only one direction
            edge_set.add((u, v))
    
    with open(edge_file, 'w') as f:
        for u, v in sorted(edge_set):
            node_u = nodes_list[u]
            node_v = nodes_list[v]
            f.write(f"{node_u}\t{node_v}\n")
    
    # Save embeddings
    embed_file = os.path.join(output_dir, f"biosl{subnet_idx}_restored_embeddings.txt")
    np.savetxt(embed_file, embeddings, fmt='%.15f')
    
    print(f"  [Save] Restored subnet saved:")
    print(f"    - Nodes: {node_file}")
    print(f"    - Edges: {edge_file}")
    print(f"    - Embeddings: {embed_file}")


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
    parser.add_argument('--restore_ratio', type=float, default=0.001,
                       help='Ratio of removed edges to restore (0-1)')
    parser.add_argument('--original_ppi', type=str, 
                       default='D:/#postgraduate/PCI/CODE/Construct subset/dataset/biogrid.txt',
                       help='Path to original PPI network file')
    parser.add_argument('--go_attr_file', type=str,
                       default='D:/#postgraduate/PCI/CODE/Construct subset/protein_go_with_activity.txt',
                       help='Path to protein GO annotation file')
    parser.add_argument('--early_stopping', action='store_true', 
                       help='Enable early stopping mechanism')
    parser.add_argument('--patience', type=int, default=50,
                       help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=1e-6,
                       help='Minimum change to qualify as improvement')
    parser.add_argument('--output_dir', type=str, default='restored_subnets',
                       help='Directory to save restored subnet files')
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
    path = osp.expanduser('D:/#postgraduate/PCI/CODE/Construct subset/dataset')
    path = osp.join(path, args.dataset)

    for ii in range(255):
        attr_file = f"D:/#postgraduate/PCI/CODE/Construct subset/biosl/Attribute_biosl{ii}.txt"
        network_file = f"D:/#postgraduate/PCI/CODE/Construct subset/biosl/Network_biosl{ii}.txt"
        node_file = f"D:/#postgraduate/PCI/CODE/Construct subset/biosl/biosl{ii}_node.txt"
        subnet_edge_file = f"D:/#postgraduate/PCI/CODE/Construct subset/biosl/biogrid_{ii}.txt"

        if not os.path.exists(attr_file) or not os.path.exists(network_file):
            continue

        print(f"\n{'='*60}")
        print(f"Processing subnet {ii}")
        print(f"{'='*60}")

        # Load original attribute matrix and extract GO terms used
        A = np.loadtxt(attr_file, delimiter=" ")
        
        # Extract GO terms from the attribute file header or reconstruct from data
        # Assuming attribute matrix columns correspond to GO terms in sorted order
        # We need to get these GO terms to maintain consistency
        attr_go_terms = []
        with open(args.go_attr_file, 'r') as f:
            all_go_set = set()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    terms_field = parts[1]
                    go_terms = terms_field.strip().split()
                    for term in go_terms:
                        if term.startswith("GO:"):
                            all_go_set.add(term.strip())
        attr_go_terms = sorted(all_go_set)
        
        # Handle attribute matrix format
        if A.ndim == 2 and A.shape[1] > 0:
            # Check if first column is index
            if A.shape[1] > len(attr_go_terms):
                A = A[:, 1:]  # Remove index column
        
        N = np.loadtxt(network_file, delimiter=" ")
        edge_index = torch.tensor(np.array(np.nonzero(N)), dtype=torch.long)
        x = torch.tensor(A, dtype=torch.float)
        
        print(f"Original: nodes={x.shape[0]}, edges={edge_index.shape[1]//2}, features={x.shape[1]}")
        
        # Load removed edges and nodes
        removed_edges = set()
        removed_nodes = set()
        node2idx = {}
        protein_go = {}
        subnet_nodes_list = []
        
        if os.path.exists(node_file) and os.path.exists(subnet_edge_file):
            removed_edges, removed_nodes, subnet_nodes_list, node2idx, protein_go = \
                load_removed_edges_and_nodes(
                    original_ppi_file=args.original_ppi,
                    subnet_edge_file=subnet_edge_file,
                    node_file=node_file,
                    go_attr_file=args.go_attr_file
                )
        
        # Restore edges and nodes randomly
        restored_nodes_list = subnet_nodes_list
        if len(removed_edges) > 0 and args.restore_ratio > 0:
            edge_index, x, node2idx, restored_nodes_list, num_restored_edges, num_restored_nodes = \
                restore_edges_and_nodes(
                    edge_index, x, removed_edges, removed_nodes,
                    subnet_nodes_list, node2idx, protein_go, attr_go_terms,
                    restore_ratio=args.restore_ratio
                )
            print(f"After restoration: nodes={x.shape[0]}, edges={edge_index.shape[1]//2}")
        else:
            print(f"  [Skip] No edges to restore or restore_ratio=0")
        
        # Create data object
        from torch_geometric.data import Data
        data = Data(edge_index=edge_index, x=x)
        data = data.to(device)

        # Create model - use augmented adjacency matrix
        N_augmented = np.zeros((x.shape[0], x.shape[0]), dtype=int)
        edge_list = edge_index.cpu().numpy()
        for i in range(edge_list.shape[1]):
            u, v = edge_list[0, i], edge_list[1, i]
            N_augmented[u, v] = 1

        encoder = Encoder(data.num_features, num_hidden, get_activation(activation),
                         base_model=GConv, k=num_layers).to(device)
        model = KEGCL(encoder, N_augmented, num_hidden, num_proj_hidden, tau).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        log = args.verbose.split(',')

        # Initialize early stopping
        early_stopping = None
        if args.early_stopping:
            early_stopping = EarlyStopping(
                patience=args.patience, 
                min_delta=args.min_delta,
                verbose='train' in log
            )
            print(f"Early stopping enabled (patience={args.patience})")

        # Training loop
        stopped_early = False
        final_epoch = num_epochs
        
        for epoch in range(1, num_epochs + 1):
            loss = train(model, optimizer, data, drop_edge_rate_1, drop_edge_rate_2,
                        drop_feature_rate_1, drop_feature_rate_2)
            
            if 'train' in log and epoch % 100 == 0:
                print(f'  Epoch={epoch:03d}, loss={loss:.4f}')
            
            # Check early stopping
            if early_stopping is not None:
                if early_stopping(loss, epoch):
                    stopped_early = True
                    final_epoch = epoch
                    break
        
        # Training completed
        print(f"\nSubnet {ii} training completed:")
        print(f"  Total epochs: {final_epoch}/{num_epochs}")
        if stopped_early:
            print(f"  Status: Early stopped (best_epoch={early_stopping.best_epoch})")
        else:
            print(f"  Status: Completed normally")
        
        # Generate final embeddings for restored subnet
        model.eval()
        with torch.no_grad():
            z_final = model(data.x, data.edge_index, [2, 2], final=True).detach().cpu().numpy()
        
        # Save restored subnet files
        save_restored_subnet(ii, restored_nodes_list, edge_index, z_final, args.output_dir)
        
        print(f"{'='*60}\n")

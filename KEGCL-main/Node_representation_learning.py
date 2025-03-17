import argparse
import os.path as osp
import random
import nni
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
from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE, NewGConv, NewEncoder, NewGRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from utils import normalize_adjacency_matrix, create_adjacency_matrix, load_adj_neg, Rank

def train():
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0] #adjacency with edge droprate 2
    x_1 = drop_feature(data.x, drop_feature_rate_1)#3
    x_2 = drop_feature(data.x, drop_feature_rate_2)#4
    z1 = model(x_1, edge_index_1, [2, 2])
    z2 = model(x_2, edge_index_2, [8, 8])

    loss = model.loss(z1, z2, batch_size=64)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(final=False):

    model.eval()
    z = model(data.x, data.edge_index, [1, 1], final=True)

    evaluator = MulticlassEvaluator()
    acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=0)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='BIO')
    parser.add_argument('--config', type=str, default='param.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(args.seed)
    use_nni = args.config == 'nni'
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

    device = torch.device(args.device)
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    adj = 0

    A = np.loadtxt("ppi/ABIOnew_cleaned.txt", delimiter=" ")[:, 1:]  # 节点属性矩阵
    N = np.loadtxt("ppi/Network_BIOnew.txt", delimiter=" ")  # 加载邻接矩阵
    edge_index = torch.tensor(np.array(np.nonzero(N)), dtype=torch.long)

    x = torch.tensor(A, dtype=torch.float)
    from torch_geometric.data import Data
    data = Data(edge_index=edge_index, x=x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    encoder = NewEncoder(data.num_features, num_hidden, get_activation(activation),
                      base_model=NewGConv, k=num_layers).to(device)

    model = NewGRACE(encoder, adj, num_hidden, num_proj_hidden, tau).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )   

    log = args.verbose.split(',')

    for epoch in range(1, num_epochs + 1):

        loss = train()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        if epoch % 100 == 0:
            acc = test()
            x_1 = drop_feature(data.x, drop_feature_rate_1)
            x_2 = drop_feature(data.x, drop_feature_rate_2)
            x_3 = drop_feature(sub_x, drop_feature_rate_1)

            edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
            edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
            z = model(data.x, data.edge_index, [2, 2], final=True).detach().cpu().numpy()
            z1 = model(x_1, edge_index_1, [2, 2], final=True).detach().cpu().numpy()
            z2 = model(x_2, edge_index_2, [2, 2], final=True).detach().cpu().numpy()
            np.savetxt("dipnew_embed.txt", z.detach().cpu().numpy(), fmt='%.15f')
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')



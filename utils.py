import os
import torch
import random
import numpy as np
import networkx as nx
import scipy.sparse as ssp
import torch_geometric as tg
from tqdm import tqdm
from torch_geometric.data import Data
from scipy.sparse import csc_matrix

device = torch.device('cuda:0')


def move_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loader(data_name='enron10'):
    data_root = './data/'
    filepath = data_root + '{}.data'.format(data_name)
    data = torch.load(filepath)
    assert data_name in ['enron10', 'dblp', 'as733', 'fbw', 'enron', 'HepPh30', 'uci', 'yelp', 'ml-10m', 'COLAB']
    test_len = None
    trainable_feat = None
    if data_name == 'enron10':
        test_len = 3
    elif data_name == 'dblp':
        test_len = 3
    elif data_name == 'as733':
        test_len = 10
    elif data_name == 'fbw':
        test_len = 3
    elif data_name == 'HepPh30':
        test_len = 6
    if data_name == 'enron':
        test_len = 3
    if data_name == 'uci':
        test_len = 3
    if data_name == 'yelp':
        test_len = 3
    if data_name == 'ml-10m':
        test_len = 3
    if data_name == 'COLAB':
        test_len = 3
    return data, test_len, trainable_feat


def split_data(data, t, test_ratio, is_new):
    if is_new:
        pos_index = data['new_pedges'][t].long().numpy()
        neg_index = data['new_nedges'][t].long().numpy()
    else:
        pos_index = data['pedges'][t].long().numpy()
        neg_index = data['nedges'][t].long().numpy()

    train_pos = pos_index
    train_neg = neg_index

    return train_pos, train_neg


def get_test_data(data, test_shots, is_new, is_multi):
    if is_multi:
        pos_list = []
        neg_list = []
        for t in test_shots:
            if is_new:
                pos_index = data['new_pedges'][t].long().numpy()
                neg_index = data['new_nedges'][t].long().numpy()
            else:
                pos_index = data['pedges'][t].long().numpy()
                neg_index = data['nedges'][t].long().numpy()
            # assert pos_index.shape[1] == neg_index.shape[1]
            # num_edges = pos_index.shape[1]
            # num_samples = min(pos_index.shape[1], 1000)
            # seed = [k for k in range(num_edges)]
            # samples = random.sample(seed, num_samples)
            # pos_list.append(pos_index[:, samples])
            # neg_list.append(neg_index[:, samples])
            pos_list.append(pos_index)
            neg_list.append(neg_index)
        test_pos = np.concatenate(pos_list, axis=1)
        test_neg = np.concatenate(neg_list, axis=1)
    else:
        t = test_shots[0]
        if is_new:
            pos_index = data['new_pedges'][t].long().numpy()
            neg_index = data['new_nedges'][t].long().numpy()
        else:
            pos_index = data['pedges'][t].long().numpy()
            neg_index = data['nedges'][t].long().numpy()
        # assert pos_index.shape[1] == neg_index.shape[1]
        # num_edges = pos_index.shape[1]
        # num_samples = min(pos_index.shape[1], 1000)
        # seed = [k for k in range(num_edges)]
        # samples = random.sample(seed, num_samples)
        # test_pos = pos_index[:, samples]
        # test_neg = neg_index[:, samples]
        test_pos = pos_index
        test_neg = neg_index
    return test_pos, test_neg


def get_train_graph(data, train_shots):
    graph_list = []
    edge_index_list = []
    for t in train_shots:
        edge_index_list.append(data['edge_index_list'][t].long())
        edge_index = data['edge_index_list'][t].long().numpy()
        src = edge_index[0]
        dst = edge_index[1]
        weights = np.array([1.0] * edge_index.shape[1], dtype=float)
        graph = csc_matrix((weights, (src, dst)), shape=(data['num_nodes'], data['num_nodes']))
        graph_list.append(graph)
    all_edge = torch.cat(edge_index_list, dim=1).T.tolist()
    collapsed_graph = nx.Graph()
    collapsed_graph.add_nodes_from([k for k in range(data['num_nodes'])])
    collapsed_graph.add_edges_from(all_edge)
    collapsed_edge = torch.tensor(list(collapsed_graph.edges()), dtype=torch.long).T
    collapsed_edge = torch.cat([collapsed_edge, collapsed_edge[[1, 0]]], dim=1)
    collapsed_src = collapsed_edge[0]
    collapsed_dst = collapsed_edge[1]
    collapsed_weights = np.array([1.0] * collapsed_edge.shape[1], dtype=float)
    collapsed_graph = csc_matrix((collapsed_weights, (collapsed_src, collapsed_dst)),
                                 shape=(data['num_nodes'], data['num_nodes']))
    graph_list.append(collapsed_graph)
    return graph_list


def links_to_subgraphs(graph_list, train_pos, train_neg, test_pos, test_neg, hop):
    train_subgraphs_list = []
    test_subgraphs_list = []
    max_num_node_labels_list = []
    for graph in graph_list:
        train_pos_subgraphs, train_pos_max_num_node_labels = links2subgraphs(graph, train_pos, hop, 1)
        train_neg_subgraphs, train_neg_max_num_node_labels = links2subgraphs(graph, train_neg, hop, 0)
        test_pos_subgraphs, test_pos_max_num_node_labels = links2subgraphs(graph, test_pos, hop, 1)
        test_neg_subgraphs, test_neg_max_num_node_labels = links2subgraphs(graph, test_neg, hop, 0)
        train_subgraphs = train_pos_subgraphs + train_neg_subgraphs
        test_subgraphs = test_pos_subgraphs + test_neg_subgraphs
        max_num_node_labels = max(
            [train_pos_max_num_node_labels,
             train_neg_max_num_node_labels,
             test_pos_max_num_node_labels,
             test_neg_max_num_node_labels])
        train_subgraphs_list.append(train_subgraphs)
        test_subgraphs_list.append(test_subgraphs)
        max_num_node_labels_list.append(max_num_node_labels)
    return train_subgraphs_list, test_subgraphs_list, max(max_num_node_labels_list)


def links2subgraphs(graph, links, hop, label):
    subgraph_list = []
    max_num_node_labels = 0
    # for i, j in tqdm(zip(links[0], links[1])):
    for i, j in zip(links[0], links[1]):
        subgraph, node_labels = subgraph_extraction_labeling((i, j), graph, hop)
        max_num_node_labels = max(max_num_node_labels, max(node_labels))
        node_labels = torch.tensor(node_labels, dtype=torch.long).unsqueeze(dim=0).T
        adj = nx.adjacency_matrix(subgraph)
        adj_triu = ssp.triu(adj)
        edge_index, _ = tg.utils.from_scipy_sparse_matrix(adj_triu)
        pyg_subgraph = Data(x=None, edge_index=edge_index, node_labels=node_labels, subgraph_label=label)
        subgraph_list.append(pyg_subgraph)
    return subgraph_list, max_num_node_labels


def subgraph_extraction_labeling(link, graph, hop):
    nodes = set(list([link[0], link[1]]))
    visited = set(list([link[0], link[1]]))
    fringe = set(list([link[0], link[1]]))
    nodes_dist = [0, 0]
    for dist in range(1, hop + 1):
        fringe = neighbors(fringe, graph)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) > 100:
            fringe = random.sample(fringe, 100)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    nodes.remove(link[0])
    nodes.remove(link[1])
    nodes = [link[0], link[1]] + list(nodes)
    subgraph = graph[nodes, :][:, nodes]
    node_labels = DRNL(subgraph)
    g = nx.from_scipy_sparse_array(subgraph)
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, node_labels


def neighbors(fringe, A):
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def DRNL(subgraph):
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0] + list(range(2, K)), :][:, [0] + list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    where_inf_0 = np.isinf(dist_to_0)
    where_inf_1 = np.isinf(dist_to_1)
    dist_to_0[where_inf_0] = 1e6
    dist_to_1[where_inf_1] = 1e6
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0
    labels[labels < -1e6] = 0
    return labels


def to_hypergraphs(graphs_list, max_num_node_labels, batch, shuffle):
    seeds = [k for k in range(len(graphs_list[0]))]
    if shuffle:
        random.shuffle(seeds)
    batch_hypergraphs_list = []
    for graphs in graphs_list:
        batch_hypergraphs = to_batch(graphs, max_num_node_labels, batch, seeds)
        batch_hypergraphs_list.append(batch_hypergraphs)
    batch_hypergraphs_list_T = []
    for i in range(len(batch_hypergraphs_list[0])):
        batch_hypergraphs_T = []
        for j in range(len(batch_hypergraphs_list)):
            batch_hypergraphs_T.append(batch_hypergraphs_list[j][i])
        batch_hypergraphs_list_T.append(batch_hypergraphs_T)

    return batch_hypergraphs_list_T


def to_batch(graphs, max_num_node_labels, batch, seeds):
    batch_hypergraphs = []
    batch_node_feats = []
    batch_edge_feats = []
    batch_edges = []
    batch_labels = []
    batch_marks = []
    batch_node_marks = []
    b = 1
    i = 0
    mark = 0
    bias = 0
    for seed in seeds:
        i += 1
        node_feats = labels_to_feats(graphs[seed].node_labels, max_num_node_labels)
        edge_feats = get_edge_feats(node_feats, graphs[seed].edge_index)
        batch_node_feats.append(node_feats)
        batch_edge_feats.append(edge_feats)
        batch_edges.append(graphs[seed].edge_index + bias)
        batch_labels.append(graphs[seed].subgraph_label)
        batch_marks.append(mark)
        batch_node_marks.append(bias)
        if b == batch or i == len(graphs):
            tensor_batch_node_feats = torch.cat(batch_node_feats, dim=0)
            tensor_batch_edge_feats = torch.cat(batch_edge_feats, dim=0)
            tensor_batch_edges = torch.cat(batch_edges, dim=1)
            tensor_batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            tensor_batch_marks = torch.tensor(batch_marks, dtype=torch.long)
            tensor_batch_node_marks = torch.tensor(batch_node_marks, dtype=torch.long)
            tensor_batch_hyperedges = DHT(tensor_batch_edges)
            batch_hypergraph = Data(x=tensor_batch_edge_feats,
                                    edge_index=tensor_batch_hyperedges,
                                    y=tensor_batch_labels,
                                    marks=tensor_batch_marks,
                                    edge_attr=tensor_batch_node_feats,
                                    edge_marks=tensor_batch_node_marks
                                    )
            batch_hypergraphs.append(batch_hypergraph)
            batch_node_feats = []
            batch_edge_feats = []
            batch_edges = []
            batch_labels = []
            batch_marks = []
            batch_node_marks = []
            b = 1
            mark = 0
            bias = 0
        else:
            b += 1
            mark += graphs[seed].edge_index.shape[1]
            bias += graphs[seed].node_labels.shape[0]
    return batch_hypergraphs


def labels_to_feats(labels, max_labels):
    feats = torch.zeros(labels.shape[0], max_labels + 1)
    feats.scatter_(1, labels, 1).to(torch.float)
    return feats


def get_edge_feats(node_feats, edges):
    src = edges[0]
    tar = edges[1]
    src_feats = node_feats[src]
    tar_feats = node_feats[tar]
    edge_feats = torch.cat([torch.min(src_feats, tar_feats), torch.max(src_feats, tar_feats)], dim=1)
    return edge_feats


def DHT(edge_index):
    num_edge = edge_index.size(1)
    device = edge_index.device

    hyperedge_ids = torch.arange(num_edge, device=device).repeat_interleave(2)
    node_indices = edge_index.T.flatten()
    hyperedge_index = torch.stack([hyperedge_ids, node_indices], dim=0).long()

    return hyperedge_index

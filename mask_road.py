from model import TraGraph, GCN, construct_negative_graph, compute_loss
from utils import init_env, calculate_pairwise_connectivity

import time
import torch
import torch.nn as nn
import dgl
import os
import random
import numpy as np
import networkx as nx
import argparse

parser = argparse.ArgumentParser(description='elec attack')

parser.add_argument('--epoch', type=int, default=1000, help='Times to train')
parser.add_argument('--batch', type=int, default=20, help='Data number used to train')
parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
parser.add_argument('--lr', type=float, default=0.01, help='Laerning rate')
parser.add_argument('--epsilon', type=float, default=0.6, help='Epsilon greedy policy')

args = parser.parse_args()

# tpt = './embedding/primary.pt'
tpt = './embedding/tra_feat.pt'
perturb_pth = './embedding/perturb_primary.pt'
# perturb_pth = './embedding/perturb_ter.pt'

TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
EPOCH = args.epoch
LR = args.lr
BATCH_SIZE = args.batch
GAMMA = args.gamma
EPSILON = args.epsilon
MEMORY_CAPACITY = 1000
TARGET_REPLACE_ITER = 25

random.seed(20230124)
device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    tgraph = TraGraph(file1=TFILE1, file2=TFILE2, file3=TFILE3,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    r_type='primary',
                    khop=KHOP,
                    epochs=300,
                    pt_path=tpt)

    node_num = tgraph.node_num

    tnodes = list(tgraph.nxgraph.nodes())
    tedges = list(tgraph.nxgraph.edges())

    random.shuffle(tedges)
    tedges = tedges[:-10]

    perturb_graph = nx.Graph()
    perturb_graph.add_nodes_from(tnodes)
    perturb_graph.add_edges_from(tedges)

    degree = dict(nx.degree(perturb_graph))
    degree_list = sorted(degree.items(), key = lambda x:x[1], reverse=True)[:20]
    tgc = perturb_graph.copy()
    orig_val = calculate_pairwise_connectivity(tgc)
    result = []
    for id, (node, degree) in enumerate(degree_list):
            tgc.remove_node(node)
            val = calculate_pairwise_connectivity(tgc) / orig_val
            result.append([id+1, val])

    result = np.array(result)
    np.savetxt('./results/mask_road_degree.txt', result)

    CI = []
    degree = dict(nx.degree(perturb_graph))
    for node in degree:
        ci = 0
        neighbors = list(perturb_graph.neighbors(node))
        for neighbor in neighbors:
            ci += (degree[neighbor]-1)
        CI.append((node,ci*(degree[node]-1)))

    result = []
    ci_list = sorted(CI, key = lambda x:x[1],reverse = True)[:20]
    tgc = tgraph.nxgraph.copy()
    for id, (node, CI) in enumerate(ci_list):
        tgc.remove_node(node)
        val = calculate_pairwise_connectivity(tgc) / orig_val
        result.append([id+1, val])

    result = np.array(result)
    np.savetxt('./results/mask_road_ci.txt', result)

    perturb_graph = dgl.from_networkx(perturb_graph)
    orig_feat = tgraph.feat.detach()
    
    perturb_graph.ndata['feat'] = orig_feat.clone().requires_grad_()

    gcn = GCN(EMBED_DIM, HID_DIM, FEAT_DIM)
    optimizer = torch.optim.Adam(gcn.parameters())
    optimizer.zero_grad()

    for epoch in range(500):
        t = time.time()
        negative_graph = construct_negative_graph(perturb_graph, 5)
        pos_score, neg_score = gcn(perturb_graph, negative_graph, perturb_graph.ndata['feat'])
        feat = gcn.sage(perturb_graph, perturb_graph.ndata['feat'])
        dist = torch.dist(feat, orig_feat, p=2)
        
        loss = compute_loss(pos_score, neg_score) + dist * 10
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                        " time=", "{:.4f}s".format(time.time() - t)
                        )
        
    print(" train_loss = ", "{:.5f} ".format(loss.item()))

    feat = gcn.sage(perturb_graph, perturb_graph.ndata['feat'])
    torch.save(feat, perturb_pth)

    
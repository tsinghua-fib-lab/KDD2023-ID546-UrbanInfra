from model import ElecGraph, TraGraph, Bigraph, HeteroGCN
from model import construct_negative_graph_with_type, compute_loss, numbers_to_etypes
from utils import init_env, nodes_ranked_by_CI, nodes_ranked_by_Degree, calculate_pairwise_connectivity
from utils import influenced_tl_by_elec

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

FILE = './data/electricity/e10kv2tl.json'
EFILE = './data/electricity/all_dict_correct.json'
TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
ept = './embedding/elec_feat.pt'
tpt = './embedding/tra_feat.pt'
bpt = ('./embedding/bifeatures/bi_elec_feat.pt', './embedding/bifeatures/bi_tra_feat.pt') 
perturb_bpt = ('./embedding/bifeatures/p_bi_elec_feat.pt', './embedding/bifeatures/p_bi_tra_feat.pt') 
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
EPOCH = args.epoch
LR = args.lr
BATCH_SIZE = args.batch
GAMMA = args.gamma
EPSILON = args.epsilon
MEMORY_CAPACITY = 2000
TARGET_REPLACE_ITER = 25
BASE = 100000000
MAX_DP = 660225576
NUM_TRAIN = 30
NUM_TEST = 20


device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    egraph = ElecGraph(file=EFILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=500,
                    pt_path=ept)

    tgraph = TraGraph(file1=TFILE1, file2=TFILE2, file3=TFILE3,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    r_type='tertiary',
                    khop=KHOP,
                    epochs=300,
                    pt_path=tpt)

    bigraph = Bigraph(efile=EFILE, tfile1=TFILE1, tfile2=TFILE2, tfile3=TFILE3, file=FILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    r_type='tertiary',
                    subgraph = (egraph, tgraph),
                    khop=KHOP,
                    epochs=600,
                    pt_path=bpt)
    
    n_power = egraph.node_num
    n_junc  = tgraph.node_num
    power_idx = {v:k for k, v in egraph.node_list.items()}
    junc_idx  = {v:k-n_power for k, v in tgraph.node_list.items()}

    
    edge_list = bigraph.nxgraph.edges()
    elec_edge = [(u, v) for (u, v) in edge_list if u < 6e8 and v < 6e8]
    tran_edge = [(u, v) for (u, v) in edge_list if u > 6e8 and v > 6e8]
    supp_edge = [(u, v) for (u, v) in edge_list 
                            if (u, v) not in elec_edge and (u, v) not in tran_edge]

    # perturbation
    random.shuffle(tran_edge)
    tran_edge = tran_edge[:-10]     # 5205 --> 5195

    elec_src, elec_dst = np.array([power_idx[u] for (u, _) in elec_edge]), np.array([power_idx[v] for (_, v) in elec_edge])
    tran_src, tran_dst = np.array([junc_idx[u] for (u, _) in tran_edge]), np.array([junc_idx[v] for (_, v) in tran_edge])
    supp_src, supp_dst = np.array([junc_idx[u] for (u, _) in supp_edge]), np.array([power_idx[v] for (_, v) in supp_edge])
    
    txgraph = dgl.graph((tran_src, tran_dst))
    txgraph = dgl.to_networkx(txgraph).to_undirected()

    hetero_graph = dgl.heterograph({
                ('power', 'elec', 'power'): (elec_src, elec_dst),
                ('power', 'eleced-by', 'power'): (elec_dst, elec_src),
                ('junc', 'tran', 'junc'): (tran_src, tran_dst),
                ('junc', 'traned-by', 'junc'): (tran_dst, tran_src),
                ('junc', 'supp', 'power'): (supp_src, supp_dst),
                ('power', 'suppd-by', 'junc'): (supp_dst, supp_src)
                })
    
    egraph_degree = nodes_ranked_by_Degree(egraph.nxgraph)[:10]
    egraph_CI = nodes_ranked_by_CI(egraph.nxgraph)[:10]

    tgraph_degree = nodes_ranked_by_Degree(txgraph)[:10]
    tgraph_CI = nodes_ranked_by_CI(txgraph)[:10]

    tgraph_degree = [tgraph.node_list[node+n_power] for node in tgraph_degree]
    tgraph_CI = [tgraph.node_list[node+n_power] for node in tgraph_CI]

    attack_nodes_degree = egraph_degree + tgraph_degree
    random.shuffle(attack_nodes_degree)

    attack_nodes_CI = egraph_CI + tgraph_CI
    random.shuffle(attack_nodes_CI)

    np.savetxt('./result/test/bi_degree_nodes.txt', attack_nodes_degree)
    np.savetxt('./result/test/bi_CI_nodes.txt', attack_nodes_CI)

    tgc = txgraph.copy()
    result = [0]
    elec_env = init_env()
    origin_val = calculate_pairwise_connectivity(tgc)
    t_val = 1
    tpower = elec_env.ruin([])
    total_reward = 0
    choosen_road = []
    choosen_elec = []

    power_record = []
    gcc_record = []

    for node in attack_nodes_degree:
        h_val = t_val
        hpower = tpower

        if node //BASE == 9:
            choosen_road.append(node)
        else:
            choosen_elec.append(node)
            
        tpower,elec_state = elec_env.ruin(choosen_elec,flag=0)
        choosen_road += influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
        tgc.remove_nodes_from(choosen_road)
        t_val = calculate_pairwise_connectivity(tgc) / origin_val

        reward_elec = (hpower - tpower) / MAX_DP
        reward_road = (h_val - t_val) 
        reward = (reward_road + reward_elec) * 1e4
        total_reward += reward

        power_record.append(tpower)
        gcc_record.append(t_val)
        result.append(total_reward)

    print("degree attack power: ", power_record)
    print("degree attack road: ", gcc_record)
    np.savetxt('./result/test/mask_bi_result_degree.txt', result)

    tgc = txgraph.copy()
    result = [0]
    elec_env = init_env()
    origin_val = calculate_pairwise_connectivity(tgc)
    t_val = 1
    tpower = elec_env.ruin([])
    total_reward = 0
    choosen_road = []
    choosen_elec = []

    power_record = []
    gcc_record = []

    for node in attack_nodes_CI:
        h_val = t_val
        hpower = tpower

        if node //BASE == 9:
            choosen_road.append(node)
        else:
            choosen_elec.append(node)
            
        tpower,elec_state = elec_env.ruin(choosen_elec,flag=0)
        choosen_road += influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
        tgc.remove_nodes_from(choosen_road)
        t_val = calculate_pairwise_connectivity(tgc) / origin_val

        reward_elec = (hpower - tpower) / MAX_DP
        reward_road = (h_val - t_val) 
        reward = (reward_road + reward_elec) * 1e4
        total_reward += reward

        power_record.append(tpower)
        gcc_record.append(t_val)
        result.append(total_reward)

    print("CI attack power: ", power_record)
    print("CI attack road: ", gcc_record)
    np.savetxt('./result/test/mask_bi_result_CI.txt', result)


    hetero_graph.nodes['power'].data['feature'] = torch.nn.Embedding(n_power, EMBED_DIM, max_norm=1).weight
    hetero_graph.nodes['junc'].data['feature'] = torch.nn.Embedding(n_junc, EMBED_DIM, max_norm=1).weight
    
    hgcn = HeteroGCN(EMBED_DIM, HID_DIM, FEAT_DIM, hetero_graph.etypes)
    
    bifeatures = {
            'junc' :bigraph.feat['junc'].clone().requires_grad_(),
            'power':bigraph.feat['power'].clone().requires_grad_()
    }

    orig_feat = bigraph.feat
    orig_feat = torch.concatenate((orig_feat['power'], orig_feat['junc']), dim = 0)
    orig_feat = orig_feat.detach()
    # print(orig_feat.shape)      [15712, 64]
    
    optimizer = torch.optim.Adam(hgcn.parameters())
    print('training features ...')
    for epoch in range(600):

        num = epoch % 6
        etype = numbers_to_etypes(num)

        t = time.time()
        negative_graph = construct_negative_graph_with_type(hetero_graph, 5, etype)
        pos_score, neg_score = hgcn(hetero_graph, negative_graph, bifeatures, etype)
        
        feat = torch.concatenate((bifeatures['power'], bifeatures['junc']), dim = 0)
        dist = torch.dist(feat, orig_feat, p=2)
        
        loss = compute_loss(pos_score, neg_score) + dist*10
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                                " time=", "{:.4f}s".format(time.time() - t)
                                )

    feat = hgcn.layer(hetero_graph, bifeatures)
    try:
        torch.save(feat['power'], perturb_bpt[0])
        torch.save(feat['junc'], perturb_bpt[1])
        print("saving features sucess")
    except:
        print("saving features failed")
import argparse
import time

import numpy as np
import torch
import random
from colorama import Back, Fore, Style, init
from model import DQN, Bigraph, ElecGraph, TraGraph
from utils import (calculate_pairwise_connectivity, influenced_tl_by_elec,
                   init_env, nodes_ranked_by_CI, nodes_ranked_by_Degree)

FILE = './data/e10kv2tl.json'
EFILE = './data/electricity/all_dict_correct.json'
TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
ept = './embedding/elec_feat.pt'
tpt = './embedding/tra_feat.pt'
bpt = ('./embedding/bifeatures/bi_elec_feat.pt', './embedding/bifeatures/bi_tra_feat.pt') 
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
MAX_DP = 660225576
BASE = 100000000

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
                    khop=KHOP,
                    epochs=300,
                    pt_path=tpt)
    bigraph = Bigraph(efile=EFILE, tfile1=TFILE1, tfile2=TFILE2, tfile3=TFILE3, file=FILE,
                embed_dim=EMBED_DIM,
                hid_dim=HID_DIM,
                feat_dim=FEAT_DIM,
                subgraph = (egraph, tgraph),
                khop=KHOP,
                epochs=600,
                pt_path=bpt)


    num_elec = 10
    num_road = 10
    bigraph_CI = nodes_ranked_by_CI(bigraph.nxgraph)
    bigraph_degree = nodes_ranked_by_Degree(bigraph.nxgraph)
    egraph_CI = nodes_ranked_by_CI(egraph.nxgraph)
    egraph_degree = nodes_ranked_by_Degree(egraph.nxgraph)
    tgraph_CI = nodes_ranked_by_CI(tgraph.nxgraph)
    tgraph_degree = nodes_ranked_by_Degree(tgraph.nxgraph)



    # attack_nodes = egraph_CI[:num_elec] + tgraph_CI[:num_road]
    attack_nodes = egraph_degree[:num_elec] + tgraph_degree[:num_road]

    # attack_nodes = bigraph_CI[:20]
    # attack_nodes = bigraph_degree[:20]
    
    random.shuffle(attack_nodes)
    
    tgc = tgraph.nxgraph.copy()
    result = [0]
    elec_env = init_env()
    origin_val = calculate_pairwise_connectivity(tgc)
    t_val = 1
    tpower = elec_env.ruin([])
    total_reward = 0
    choosen_road = []
    choosen_elec = []


    for node in attack_nodes:
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

        result.append(total_reward)

    # np.savetxt('./result/bi_degree_{}_{}.txt'.format(num_elec,num_road),np.array(result))
    # np.savetxt('./result/heuristic/bi_CI_reward_{}_{}.txt'.format(num_elec,num_road),np.array(result))
    # np.savetxt('./result/heuristic/bi_CI_nodes_{}_{}.txt'.format(num_elec,num_road),np.array(attack_nodes))
    np.savetxt('./result/heuristic/bi_degree_reward_{}_{}.txt'.format(num_elec,num_road),np.array(result))
    np.savetxt('./result/heuristic/bi_degree_nodes_{}_{}.txt'.format(num_elec,num_road),np.array(attack_nodes))

    # tgc = tgraph.nxgraph.copy()
    # reward = []
    # elec_env = init_env()
    # original_power = elec_env.ruin([])
    # origin_val = calculate_pairwise_connectivity(tgc)
    # t_val = 1
    # tpower = original_power
    # total_reward = 0
    # # for i in range(num_road):
    # #     h_val = t_val
    # #     if road_nodes_CI[i] in tgc.nodes():
    # #         tgc.remove_node(road_nodes_CI[i])
    # #     t_val = calculate_pairwise_connectivity(tgc) / origin_val
    # #     total_reward += (0.5*(h_val - t_val)*1e4)
    # #     reward.append(total_reward)
    # #     tgc = tgraph.nxgraph.copy()
    # # for i in range(num_elec):
    # #     h_val = t_val
    # #     hpower = tpower
    # #     tpower,elec_state = elec_env.ruin([elec_nodes_CI[i]],flag=0)
    # #     nodes = influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
    # #     tgc.remove_nodes_from(nodes)
    # #     t_val = calculate_pairwise_connectivity(tgc) / origin_val
    # #     total_reward += (0.5*(hpower - tpower)/1e5)
    # #     total_reward += (0.5*(h_val - t_val)*1e4)
    # #     reward.append(total_reward)
    # for i in range(num_elec):
    #     h_val = t_val
    #     hpower = tpower
    #     tpower,elec_state = elec_env.ruin([elec_nodes_rdn[i]],flag=0)
    #     nodes = influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
    #     tgc.remove_nodes_from(nodes)
    #     t_val = calculate_pairwise_connectivity(tgc) / origin_val
    #     # total_reward += (0.5*(hpower - tpower)/1e5)
    #     # total_reward += (0.5*(h_val - t_val)*1e4)
    #     reward.append(t_val)


    # np.savetxt('./result/elec_CI_{}.txt'.format(num_elec),np.array(reward))



    # # np.savetxt('./result/CI_{}_{}.txt'.format(num_elec,num_road),np.array(reward))
    # val = []
    # tgc = tgraph.nxgraph.copy()

    # origin_val = calculate_pairwise_connectivity(tgc)
    # num = 0
    # elec_env = init_env()
    # nodes3 = [node for node in egraph.nxgraph.nodes() if node//100000000 == 3]
    # for node in nodes3:
    #     tgc = tgraph.nxgraph.copy()
    # reward = []
    # elec_env = init_env()
    # original_power = elec_env.ruin([])
    # origin_val = calculate_pairwise_connectivity(tgc)
    # t_val = 1
    # tpower = original_power
    # total_reward = 0

    #     print(node)
    #     tpower,elec_state = elec_env.ruin([node],flag=0)
    #     elec_env.reset()
    #     # nodes = influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
    #     # tgc.remove_nodes_from(nodes)
    #     # result = calculate_pairwise_connectivity(tgc) / origin_val
    #     val.append((node,tpower))
    #     # val.append((node,result))

    # val = sorted(val,key =lambda x:x[1],reverse = True)
    # # np.savetxt('best_cascade_node.txt',np.array(val))
    # np.savetxt('best_cascade_node_power.txt',np.array(val))




    
    


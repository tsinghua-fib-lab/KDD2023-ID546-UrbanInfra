from model import ElecGraph, TraGraph, Bigraph, Regressor
from utils import init_env, influenced_tl_by_elec, calculate_pairwise_connectivity

import torch
import random
import numpy as np
from tqdm import tqdm

FILE = './data/electricity/e10kv2tl.json'
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

random.seed(20230125)
device = torch.device("cuda:1" if torch.cuda.is_available else "cpu")

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

    elec_feat = bigraph.feat['power'].detach()
    road_feat = bigraph.feat['junc'].detach()
    features = torch.vstack((elec_feat,road_feat))
    inv_dict = {id:idx for idx, id in bigraph.node_list.items()}

    elec_env = init_env()
    tgc = tgraph.nxgraph.copy()
    orig_val = calculate_pairwise_connectivity(tgc)
    threshold = 125000

    pos_samp = []
    neg_samp = []

    nodes = [node for node in list(bigraph.nxgraph.nodes()) if node//1e8 > 2]
    
    for i in tqdm(range(500), desc="Sampling: "):
        elec_env.reset()
        tgc = tgraph.nxgraph.copy()
        choosen = random.sample(nodes, 40)
        choosen_elec = [node for node in choosen if node//1e8 < 9]
        choosen_road = [node for node in choosen if node//1e8 == 9]
        
        power, elec_state = elec_env.ruin(choosen_elec, flag=0)
        choosen_road += influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
        tgc.remove_nodes_from(choosen_road)
        pwc = calculate_pairwise_connectivity(tgc) / orig_val
        reward_elec = power / MAX_DP
        reward = (reward_elec + pwc) * 1e4
        if reward > threshold:
            pos_samp.append(choosen)
        else:
            neg_samp.append(choosen)

    print("positive sample: ", len(pos_samp))
    print("negative sample: ", len(neg_samp))

    pos_samp = sum(pos_samp, [])
    neg_samp = sum(neg_samp, [])

    node_label = {}
    for node in nodes:
        pos_cnt = pos_samp.count(node)
        neg_cnt = neg_samp.count(node)
        if pos_cnt != 0 and neg_cnt != 0:
            node_label[node] = pos_cnt / (pos_cnt + neg_cnt)

    print("train data length: ", len(node_label))

    train_x = np.array([features[inv_dict[node]].numpy() for node, label in node_label.items()])
    train_y = [label for node, label in node_label.items()]

    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y).reshape(-1, 1)

    regressor = Regressor(EMBED_DIM, HID_DIM, 1)

    optimizer = torch.optim.Adam(regressor.parameters())
    loss_func = torch.nn.MSELoss()

    for epoch in range(500):

        label_pred = regressor(train_x)
        loss = loss_func(label_pred, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()))

    pred = regressor(features).flatten()
    sorted, indices = torch.sort(pred, descending=True)

    indices = [indice for indice in indices if bigraph.node_list[int(indice)]//1e8 > 2][:20]
    index = [bigraph.node_list[int(indice)] for indice in indices]

    print(index)

    result = []
    elec_env.reset()
    tgc = tgraph.nxgraph.copy()

    choosen_elec = []
    choosen_road = []
    total_reward = 0

    t_val = 1
    tpower = elec_env.ruin([])
    
    for node in index:

        h_val = t_val
        hpower = tpower
       
        if node//1e8 == 9:
            choosen_road.append(node)
        else:
            choosen_elec.append(node)

        tpower,elec_state = elec_env.ruin(choosen_elec,flag=0)
        choosen_road += influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
        tgc.remove_nodes_from(choosen_road)
        t_val = calculate_pairwise_connectivity(tgc) / orig_val

        reward_elec = (hpower - tpower) / MAX_DP
        reward_road = h_val - t_val

        reward = (reward_road + reward_elec) * 1e4
        total_reward += reward

        result.append(total_reward)

    result = np.array(result)
    np.savetxt("./results/sup_bi.txt", result)
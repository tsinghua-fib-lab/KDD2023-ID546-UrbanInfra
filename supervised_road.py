from model import TraGraph, Regressor
from utils import calculate_pairwise_connectivity

import torch
import random
import numpy as np
from tqdm import tqdm

TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
# tpt = './embedding/perturb_primary.pt'
tpt = './embedding/tra_feat.pt'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5

random.seed(20230125)
device = torch.device("cuda:1" if torch.cuda.is_available else "cpu")

if __name__ == "__main__":

    tgraph = TraGraph(file1=TFILE1, file2=TFILE2, file3=TFILE3,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    r_type='tertiary',
                    khop=KHOP,
                    epochs=300,
                    pt_path=tpt)

    features = tgraph.feat.detach()

    # print(tgraph.node_list)       index : id
    inv_dict = {id:idx for idx, id in tgraph.node_list.items()}

    tgc = tgraph.nxgraph.copy()
    initial_val = calculate_pairwise_connectivity(tgc)
    threshold = initial_val * 0.7

    pos_samp = []
    neg_samp = []

    nodes = [node for node in list(tgraph.nxgraph.nodes())]
   
    for i in tqdm(range(500), desc="Sampling: "):
        tgc = tgraph.nxgraph.copy()
        choosen = random.sample(nodes, 10)
        tgc.remove_nodes_from(choosen)
        val = calculate_pairwise_connectivity(tgc)
        if val <= threshold:
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

    for epoch in range(300):

        label_pred = regressor(train_x)
        loss = loss_func(label_pred, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()))

    pred = regressor(features).flatten()
    sorted, indices = torch.sort(pred, descending=True)

    indices = indices[:10]
    index = [tgraph.node_list[int(indice)] for indice in indices]
    
    result = []
    tgc = tgraph.nxgraph.copy()
    for node in index:
        tgc.remove_node(node)
        result.append(calculate_pairwise_connectivity(tgc) / initial_val)
   
    result = np.array(result)
    np.savetxt("./results/sup_road_ter.txt", result)
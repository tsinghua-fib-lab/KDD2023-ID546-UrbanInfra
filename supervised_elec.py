from model import ElecGraph, Regressor
from utils import init_env

import torch
import random
import numpy as np
from tqdm import tqdm

EFILE = './data/electricity/all_dict_correct.json'
ept = './embedding/elec_feat.pt'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5

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

    features = egraph.feat.detach()
    inv_dict = {id:idx for idx, id in egraph.node_list.items()}

    elec_env = init_env()
    initial_power = elec_env.ruin([])
    threshold = initial_power * 0.8

    pos_samp = []
    neg_samp = []

    nodes = [node for node in list(egraph.nxgraph.nodes()) if node//1e8 > 2 and node//1e8 < 5]
   
    for i in tqdm(range(1000), desc="Sampling: "):
        elec_env.reset()
        choosen = random.sample(nodes, 30)
        power = elec_env.ruin(choosen)
        if power <= threshold:
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

    indices = [indice for indice in indices if egraph.node_list[int(indice)]//1e8 > 2][:10]
    index = [egraph.node_list[int(indice)] for indice in indices]

    result = []
    elec_env.reset()
    for node in index:
        power = elec_env.ruin([node])
        result.append(power)

    result = np.array(result)
    np.savetxt("./results/sup_elec.txt", result)
import argparse
import time

import numpy as np
import torch
from colorama import Back, Fore, Style, init
from model import DQN, TraGraph
from utils import calculate_pairwise_connectivity

init()

parser = argparse.ArgumentParser(description='elec attack')

parser.add_argument('--epoch', type=int, default=1000, help='Times to train')
parser.add_argument('--batch', type=int, default=20, help='Data number used to train')
parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
parser.add_argument('--lr', type=float, default=0.001, help='Laerning rate')
parser.add_argument('--epsilon', type=float, default=0.8, help='Epsilon greedy policy')
parser.add_argument('--feat', type=str, required=True, help='pre-train feat or random feat')
parser.add_argument('--label', type=str, required=True, help='train or test')

args = parser.parse_args()

EFILE = './data/electricity/all_dict_correct.json'
TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
ept = './embedding/elec_feat.pt'
# tpt = './embedding/tra_feat.pt'
# tpt = './embedding/secondary.pt'
# tpt = './embedding/primary.pt'
tpt = './embedding/perturb_primary.pt'
# tpt = './embedding/perturb_ter.pt'
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

    if args.feat == "ptr":
        features = tgraph.feat.detach()
        features = features.to(device)
        MODEL_PT = './model_param/road_ptr_ter.pt'
    elif args.feat == "rdn":
        try:
            features = torch.load('./random/road_rdn_emb.pt')
        except:
            features = torch.rand(tgraph.node_num, EMBED_DIM).to(device)
            torch.save(features, './random/road_rdn_emb.pt')
        MODEL_PT = './model_param/road_rdn.pt'

    agent = DQN(in_dim=EMBED_DIM,
                hid_dim=HID_DIM,
                out_dim=EMBED_DIM,
                memory_capacity=MEMORY_CAPACITY,
                iter=TARGET_REPLACE_ITER,
                batch_size=BATCH_SIZE,
                lr=LR,
                epsilon=EPSILON,
                gamma=GAMMA,
                label=args.label,
                model_pt=MODEL_PT)

        
    print()
    print(Fore.RED,Back.YELLOW,'begin attacking ...')
    print(Style.RESET_ALL)

    if args.label == 'test':
        
        # RL attack
        tgc = tgraph.nxgraph.copy()
        origin_val = calculate_pairwise_connectivity(tgc)
        num = tgraph.node_num
        state = torch.sum(features, dim=0) / num
        choosen = []
        result = []

        done = False
        while not done:

            node = agent.attack(features, state, choosen)
            choosen.append(node)
            print(tgraph.node_list[node])
            tgc.remove_node(tgraph.node_list[node])
            num -= 1
            
            val = calculate_pairwise_connectivity(tgc) / origin_val
            _state = (state * (num+1) - features[node]) / num
            state = _state
            
            result.append([len(choosen), val])

            if len(choosen) == 20:
                done = True
        result = np.array(result)
        np.savetxt('./results/road_result_'+args.feat+'.txt', result)


        # degree attack
        result = []
        degree_list = sorted(tgraph.degree.items(), key = lambda x:x[1],reverse = True)[:20]
        tgc = tgraph.nxgraph.copy()
        for id, (node, degree) in enumerate(degree_list):
            tgc.remove_node(node)
            val = calculate_pairwise_connectivity(tgc) / origin_val
            result.append([id+1, val])

        result = np.array(result)
        np.savetxt('./results/road_degree.txt', result)

        # CI attack
        result = []
        ci_list = sorted(tgraph.CI, key = lambda x:x[1],reverse = True)[:20]
        tgc = tgraph.nxgraph.copy()
        for id, (node, CI) in enumerate(ci_list):
            tgc.remove_node(node)
            val = calculate_pairwise_connectivity(tgc) / origin_val
            result.append([id+1, val])

        result = np.array(result)
        np.savetxt('./results/road_ci.txt', result)



    elif args.label == 'train':

        for epoch in range(EPOCH):
    
            t = time.time()
            num = tgraph.node_num
            state = torch.sum(features, dim=0) / num
            total_reward = 0
            choosen = []
            exist = list(range(num))
            
            tgc = tgraph.nxgraph.copy()
            origin_val = calculate_pairwise_connectivity(tgc)

            done = False
            result = []
            while not done:

                h_val = calculate_pairwise_connectivity(tgc) / origin_val
                node = agent.choose_node(features, state, choosen, exist)
                choosen.append(node)
                # print(node)
                # print(tgraph.node_list[node])
                # print(choosen)
                tgc.remove_node(tgraph.node_list[node])
                exist.remove(node)
                num -= 1

                t_val = calculate_pairwise_connectivity(tgc) / origin_val
                _state = (state * (num+1) - features[node]) / num
                reward = (h_val - t_val) * 10000
                total_reward += reward

                agent.store_transition(state.data.cpu().numpy(),
                                        node, reward,
                                    _state.data.cpu().numpy())
                
                if agent.memory_num > agent.mem_cap:
                    
                    agent.learn(features)

                state = _state

                if len(choosen) == 20:
                    done = True
                    result.append([epoch, total_reward])
                    print("Epoch:", '%03d' % (epoch + 1), " total reward = ", "{:.5f} ".format(total_reward),
                            " time =", "{:.4f}".format(time.time() - t)
                            )

        torch.save(agent.enet.state_dict(), './model_param/road_'+args.feat+'.pt')

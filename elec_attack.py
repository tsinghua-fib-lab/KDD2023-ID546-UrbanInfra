from model import ElecGraph, TraGraph, DQN
from utils import init_env

import time
import torch
import numpy as np
import argparse

from colorama import init
from colorama import Fore,Back,Style

init()

parser = argparse.ArgumentParser(description='elec attack')

parser.add_argument('--epoch', type=int, default=1000, help='Times to train')
parser.add_argument('--batch', type=int, default=20, help='Data number used to train')
parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
parser.add_argument('--lr', type=float, default=0.01, help='Laerning rate')
parser.add_argument('--epsilon', type=float, default=0.6, help='Epsilon greedy policy')
parser.add_argument('--feat', type=str, required=True, help='pre-train feat or random feat')
parser.add_argument('--label', type=str, required=True, help='train or test')

args = parser.parse_args()

EFILE = './data/electricity/all_dict_correct.json'
# EFILE = './data/electricity/all_dict_0.json'
TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
ept = './embedding/elec_feat.pt'
# ept = './embedding/perturb_0.pt'
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

    egraph = ElecGraph(file=EFILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=500,
                    pt_path=ept)

    elec_env = init_env()

    if args.feat == "ptr":
        features = egraph.feat.detach()
        features = features.to(device)
        MODEL_PT = './model_param/elec_ptr.pt'
    elif args.feat == "rdn":
        try:
            features = torch.load('./random/elec_rdn_emb.pt')
        except:
            features = torch.rand(egraph.node_num, EMBED_DIM).to(device)
            torch.save(features, './random/elec_rdn_emb.pt')
        MODEL_PT = './model_param/elec_rdn.pt'

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

    initial_power = elec_env.ruin([])

    print()
    print(Fore.RED,Back.YELLOW,'begin attacking ...')
    print(Style.RESET_ALL)

    if args.label == 'test':
        
        # RL attack
        t = time.time()
        num = egraph.node_num
        state = torch.sum(features, dim=0) / num
        choosen = []
        elec_env.reset()

        result = []

        done = False
        while not done:
            node = agent.attack(features, state, choosen)
            
            if egraph.node_list[node]// 100000000 < 3:
                continue
            
            print(egraph.node_list[node])
            choosen.append(node)
            num -= 1

            current_power = elec_env.ruin([egraph.node_list[node]])
            _state = (state * (num+1) - features[node]) / num
            state = _state

            result.append([len(choosen), current_power])

            if len(choosen) == 10:
                done = True
        
        result = np.array(result)
        print(Fore.RED,Back.YELLOW,'saving RL attack result ...')
        print(Style.RESET_ALL)
        np.savetxt('./results/elec_result_'+args.feat+'.txt', result)

        # # degree attack
        # egraph.degree = {key:val for key, val in egraph.degree.items() if key//100000000 > 2}
        # degree_list = sorted(egraph.degree.items(), key = lambda x:x[1],reverse = True)[:10]
    
        # elec_env.reset()
        # result = []

        # for id, (node, degree) in enumerate(degree_list):
        #     current_power = elec_env.ruin([node])
        #     result.append([id+1, current_power])

        # result = np.array(result)
        # print(Fore.RED,Back.YELLOW,'saving degree attack result ...')
        # print(Style.RESET_ALL)
        # np.savetxt('./results/elec_degree.txt', result)
        
        # # CI attack
        # egraph.CI = {node:ci for node, ci in egraph.CI if node//100000000 > 2}
        # CI_list = sorted(egraph.CI.items(), key = lambda x:x[1],reverse = True)[:10]

        # elec_env.reset()
        # result = []

        # for id, (node, CI) in enumerate(CI_list):
        #     current_power = elec_env.ruin([node])
        #     result.append([id+1, current_power])

        # result = np.array(result)
        # print(Fore.RED,Back.YELLOW,'saving CI attack result ...')
        # print(Style.RESET_ALL)
        # np.savetxt('./results/elec_CI.txt', result)

    elif args.label == 'train':

        for epoch in range(EPOCH):

            t = time.time()
            num = egraph.node_num
            state = torch.sum(features, dim=0) / num
            total_reward = 0
            choosen = []
            exist = [node for node,id in egraph.node_list.items() if id//100000000 > 2]
            elec_env.reset()

            done = False
            result = []

            while not done:
                hpower = elec_env.ruin([])
                node = agent.choose_node(features, state, choosen, exist)
                if egraph.node_list[node]// 100000000 < 3:
                    continue

                choosen.append(node)
                exist.remove(node)
                num -= 1

                tpower = elec_env.ruin([egraph.node_list[node]])
                _state = (state * (num+1) - features[node]) / num

                reward = (hpower - tpower) / 1e05
                total_reward += reward

                agent.store_transition(state.data.cpu().numpy(),
                                        node, reward,
                                    _state.data.cpu().numpy())
                            
                if agent.memory_num > agent.mem_cap:
                    agent.learn(features)

                state = _state

                if len(choosen) == 100:
                    done = True
                    result.append([epoch, total_reward])
                    
                    print("Epoch:", '%03d' % (epoch + 1), " total reward = ", "{:.5f} ".format(total_reward),
                            " time =", "{:.4f}".format(time.time() - t)
                            )
                    

        torch.save(agent.enet.state_dict(), './model_param/elec_'+args.feat+'.pt')
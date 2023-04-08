import argparse
import time

import numpy as np
import torch
from colorama import Back, Fore, Style, init
from model import DQN, Bigraph, ElecGraph, TraGraph
from utils import (calculate_pairwise_connectivity, influenced_tl_by_elec,
                   init_env)

init()

parser = argparse.ArgumentParser(description='elec attack')

parser.add_argument('--epoch', type=int, default=1000, help='Times to train')
parser.add_argument('--batch', type=int, default=50, help='Data number used to train')
parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
parser.add_argument('--lr', type=float, default=0.001, help='Laerning rate')
parser.add_argument('--epsilon', type=float, default=0.7, help='Epsilon greedy policy')
parser.add_argument('--feat', type=str, required=True, help='pre-train feat or random feat')
parser.add_argument('--label', type=str, required=True, help='train or test')

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
                    pt_path=perturb_bpt)
                    
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
                model_pt='./model_param/bi_'+args.feat+'.pt')
                # model_pt='./model_param/'+args.feat+'/bi_max_reward.pt')
    
    elec_env = init_env()
    initial_power = elec_env.ruin([])

    

    if args.feat == "ptr":
        elec_feat = bigraph.feat['power'].detach()
        road_feat = bigraph.feat['junc'].detach()
        features = torch.vstack((elec_feat,road_feat))
        features = features.to(device)
        print('Load pretrained embedding')

    elif args.feat == "rdn" :
        try:
            features = np.loadtxt('./data/random_features.txt')
            features = torch.Tensor(features).to(device)
            print('Load fixed embedding')
        except:
            features = torch.rand(bigraph.node_num, EMBED_DIM).to(device)
            np.savetxt('./data/random_features.txt',features.cpu())
            print('Create new embedding')

    print()
    print(Fore.RED,Back.YELLOW,'begin attacking ...')
    print(Style.RESET_ALL)

    if args.label == 'test':
        choosen_road = []
        choosen_elec = []
        
        t = time.time()
        g = bigraph.nxgraph
        tgc = tgraph.nxgraph.copy()
        num = bigraph.node_num
        state = torch.sum(features, dim=0) / num

        total_reward = 0
        choosen = []

        elec_env.reset()
        result = []

        origin_val = calculate_pairwise_connectivity(tgc)
        t_val = calculate_pairwise_connectivity(tgc) / origin_val
        tpower = initial_power

        power_record = []
        gcc_record = []

        done = False
        while not done:

            h_val = t_val
            hpower = tpower
            node = agent.attack(features, state, choosen)
            if bigraph.node_list[node]//BASE < 3 :
                choosen.append(node)
                NUM_TEST += 1
                continue
            choosen.append(node)
            if bigraph.node_list[node]//BASE == 9:
                choosen_road.append(bigraph.node_list[node])
            else:
                choosen_elec.append(bigraph.node_list[node])
            num -= 1

            tpower,elec_state = elec_env.ruin(choosen_elec,flag=0)
            choosen_road += influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
            tgc.remove_nodes_from(choosen_road)
            t_val = calculate_pairwise_connectivity(tgc) / origin_val

            reward_elec = (hpower - tpower) / MAX_DP
            reward_road = (h_val - t_val)

            reward = (reward_road + reward_elec) * 1e4
            total_reward += reward

            _state = (state * (num+1) - features[node]) / num
            state = _state

            power_record.append(tpower)
            gcc_record.append(t_val)
            result.append(total_reward)

            if len(choosen) == NUM_TEST:
                done = True
        
        print("transfer attack power: ", power_record)
        print("transfer attack road: ", gcc_record)
        result = np.array(result)
        print(Fore.RED,Back.YELLOW,'saving RL attack result ...')
        print(Style.RESET_ALL)
        np.savetxt('./result/test/mask_bi_result_'+args.feat+'.txt', result)
        choosen =  [bigraph.node_list[node] for node in choosen]
        np.savetxt('./result/test/mask_bi_nodes_'+args.feat+'.txt', np.array(choosen))

    elif args.label == 'train':

        g = bigraph.nxgraph
        result_reward = []
        max_total_reward = 0
        for epoch in range(EPOCH):
            
            t = time.time()
            num = bigraph.node_num
            state = torch.sum(features, dim=0) / num
            total_reward = 0
            choosen = []
            choosen_road = []
            choosen_elec = []
            exist = [node for node,id in bigraph.node_list.items() if id//100000000 > 2]
            elec_env.reset()
            tgc = tgraph.nxgraph.copy()
            origin_val = calculate_pairwise_connectivity(tgc)
            t_val = calculate_pairwise_connectivity(tgc) / origin_val
            tpower = initial_power
            done = False
            result = []

            while not done:

                h_val = t_val
                hpower = tpower

                node = agent.choose_node(features, state, choosen, exist)
                if bigraph.node_list[node]//BASE < 3:
                    continue
                choosen.append(node)
                exist.remove(node)
                if bigraph.node_list[node]//BASE == 9:
                    choosen_road.append(bigraph.node_list[node])
                else:
                    choosen_elec.append(bigraph.node_list[node])
                num -= 1
                _state = (state * (num+1) - features[node]) / num

                tpower,elec_state = elec_env.ruin(choosen_elec,flag=0)
                choosen_road += influenced_tl_by_elec(elec_state, bigraph.elec2road, tgc)
                tgc.remove_nodes_from(choosen_road)
                t_val = calculate_pairwise_connectivity(tgc) / origin_val

                reward_elec = (hpower - tpower) / MAX_DP
                reward_road = (h_val - t_val) 
                reward = (reward_road + reward_elec) * 1e4
                total_reward += reward

                agent.store_transition(state.data.cpu().numpy(),
                                        node, reward,
                                    _state.data.cpu().numpy())
                
                if agent.memory_num > agent.mem_cap:
                    
                    agent.learn(features)

                state = _state

                if len(choosen) == NUM_TRAIN:
                    result_reward.append((epoch+1,total_reward))
                    done = True
                    result.append([epoch, total_reward])
                    print(Fore.RED,Back.YELLOW)
                    print("\nEpoch:", '%03d' % (epoch + 1), " total reward = ", "{:.5f} ".format(total_reward),
                            " time =", "{:.4f}".format(time.time() - t)
                            )
                    print(Style.RESET_ALL)
            if total_reward > max_total_reward:
                max_total_reward = total_reward
                torch.save(agent.enet.state_dict(), './model_param/'+args.feat+'/bi_max_reward.pt')
            if epoch % 50 == 0 or epoch>(EPOCH-10):
                torch.save(agent.enet.state_dict(), './model_param/'+args.feat+'/bi_{}.pt'.format(epoch))
                np.savetxt('./result/total_reward/bi_'+args.feat+'_reward_{}.txt'.format(epoch),np.array(result_reward))

        np.savetxt('./result/bi_'+args.feat+'_reward_{}.txt'.format(time.strftime('%d_%H_%M')),np.array(result_reward))
        torch.save(agent.enet.state_dict(), './model_param/bi_'+args.feat+'_{}.pt'.format(time.strftime('%d-%H-%M')))
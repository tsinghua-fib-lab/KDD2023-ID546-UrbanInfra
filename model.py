import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import random
import json
import time
import dgl
import dgl.nn as dnn
import dgl.function as dfn
import copy
import math
from pyproj import Geod
from shapely.geometry import Point, LineString
from pypower.api import ppoption, runpf

from colorama import init
from colorama import Fore,Back,Style

init()

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = dnn.SAGEConv(in_dim, hid_dim, 'mean')
        self.conv2 = dnn.SAGEConv(hid_dim, out_dim, 'mean')
        self.relu = nn.ReLU()

    def forward(self, graph, input):
        output = self.conv1(graph, input)
        output = self.relu(output)
        output = self.conv2(graph, output)

        return output

class RGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, rel_names):
        super().__init__()
        self.conv1 = dnn.HeteroGraphConv({
            rel: dnn.GraphConv(in_dim, hid_dim)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dnn.HeteroGraphConv({
            rel: dnn.GraphConv(hid_dim, out_dim)
            for rel in rel_names}, aggregate='sum')

        self.relu = nn.ReLU()

    def forward(self, graph, input):
        output = self.conv1(graph, input)
        output = {k: self.relu(v) for k, v in output.items()}
        output = self.conv2(graph, output)
        return output

class Innerproduct(nn.Module):
    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['feat'] = feat
            graph.apply_edges(dfn.u_dot_v('feat', 'feat', 'score'))
            return graph.edata['score']

class HeteroInnerProduct(nn.Module):
    def forward(self, graph, feat, etype):
        with graph.local_scope():
            graph.ndata['feat'] = feat
            graph.apply_edges(dfn.u_dot_v('feat', 'feat', 'score'))
            return graph.edges[etype].data['score']

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.sage = SAGE(in_dim, hid_dim, out_dim)
        self.pred = Innerproduct()

    def forward(self, graph, neg_graph, feat):
        feat = self.sage(graph, feat)
        return self.pred(graph, feat), self.pred(neg_graph, feat)

class HeteroGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, rel_names):
        super().__init__()
        self.layer = RGCN(in_dim, hid_dim, out_dim, rel_names)
        self.pred = HeteroInnerProduct()

    def forward(self, graph, neg_graph, feat, etype):
        feat = self.layer(graph, feat)
        return self.pred(graph, feat, etype), self.pred(neg_graph, feat, etype)

def numbers_to_etypes(num):
            switcher = {
                0: ('power', 'elec', 'power'),
                1: ('power', 'eleced-by', 'power'),
                2: ('junc', 'tran', 'junc'),
                3: ('junc', 'traned-by', 'junc'),
                4: ('junc', 'supp', 'power'),
                5: ('power', 'suppd-by', 'junc'),
            }

            return switcher.get(num, "wrong!")

def compute_loss(pos_score, neg_score):
        n_edges = pos_score.shape[0]

        return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

def construct_negative_graph_with_type(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


class Graph():
    def __init__(self):
        self.graph = None
        self.feat = None
        self.node_list = None

    @property
    def node_num(self):
        return self.nxgraph.number_of_nodes()

    @property
    def egde_num(self):
        return self.nxgraph.number_of_edges()

    def build_graph(self):
        pass

    def build_feat(self, embed_dim, hid_dim, feat_dim, 
                    k, epochs,
                    pt_path):

        print('training features ...')
        embedding = nn.Embedding(self.node_num, embed_dim, max_norm=1)
        self.graph.ndata['feat'] = embedding.weight

        gcn = GCN(embed_dim, hid_dim, feat_dim)
        optimizer = torch.optim.Adam(gcn.parameters())
        optimizer.zero_grad()

        for epoch in range(epochs):
            t = time.time()
            negative_graph = construct_negative_graph(self.graph, k)
            pos_score, neg_score = gcn(self.graph, negative_graph, self.graph.ndata['feat'])
            loss = compute_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                            " time=", "{:.4f}s".format(time.time() - t)
                            )

        feat = gcn.sage(self.graph, self.graph.ndata['feat'])
        try:
            torch.save(feat, pt_path)
            print("saving features sucess")
        except:
            print("saving features failed")
        return feat

class ElecGraph(Graph):
    def __init__(self, file, embed_dim, hid_dim, feat_dim, khop, epochs, pt_path):
        print(Fore.RED,Back.YELLOW)
        print('Electricity network construction!')
        print(Style.RESET_ALL)
        self.node_list, self.nxgraph,self.graph = self.build_graph(file)
        self.degree = dict(nx.degree(self.nxgraph))
        self.CI = self.build_CI()
        
        try:
            feat = torch.load(pt_path)
            print('Elec features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim, 
                                         khop, epochs,
                                         pt_path)

    def build_graph(self, file):

        print('building elec graph ...')
        try:
            elec_graph = nx.read_gpickle(file)
        except:
            with open(file, 'r') as f:
                data = json.load(f)
            elec_graph = nx.Graph()
            for key,facility in data.items():
                for node_id in facility.keys():
                    node = facility[node_id]
                    for neighbor in node['relation']:
                        if int(node_id)<6e8 and neighbor<6e8:
                            elec_graph.add_edge(int(node_id),neighbor)

        node_list : dict = {i:j for i,j in enumerate(list(elec_graph.nodes()))}
        print('electric graph builded.')
        return node_list, elec_graph, dgl.from_networkx(elec_graph)

    def build_CI(self):
        CI = []
        d = self.degree
        for node in d:
            ci = 0
            neighbors = list(self.nxgraph.neighbors(node))
            for neighbor in neighbors:
                ci += (d[neighbor]-1)
            CI.append((node,ci*(d[node]-1)))
        
        return CI

class TraGraph(Graph):
    def __init__(self, file1, file2, file3, 
                embed_dim, hid_dim, feat_dim, r_type,
                khop, epochs, pt_path):
        print(Fore.RED,Back.YELLOW)
        print('Traffice network construction!')
        print(Style.RESET_ALL)
        self.node_list, self.nxgraph, self.graph = self.build_graph(file1, file2, file3, r_type)
        self.degree = dict(nx.degree(self.nxgraph))
        self.CI = self.build_CI()
        try:
            feat = torch.load(pt_path)
            print('Traffic features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim, 
                                         khop, epochs,
                                         pt_path)

    def build_graph(self, file1, file2, file3, r_type):

        print('building traffic graph ...')
        graph = nx.Graph()
        with open(file1, 'r') as f:
            data = json.load(f)
        with open(file2, 'r') as f:
            road_type = json.load(f)
        with open(file3, 'r') as f:
            tl_id_road2elec_map = json.load(f)
        for road, junc in data.items():
            if len(junc) == 2 and road_type[road] == r_type:
                graph.add_edge(tl_id_road2elec_map[str(junc[0])], tl_id_road2elec_map[str(junc[1])])

        node_list : dict = {i:j for i,j in enumerate(list(graph.nodes()))}
        print('traffic graph builded.')
        return node_list, graph, dgl.from_networkx(graph)

    def build_CI(self):
        CI = []
        d = self.degree
        for node in d:
            ci = 0
            neighbors = list(self.nxgraph.neighbors(node))
            for neighbor in neighbors:
                ci += (d[neighbor]-1)
            CI.append((node,ci*(d[node]-1)))
        
        return CI

class Bigraph(Graph):
    def __init__(self, efile, tfile1, tfile2, tfile3, file,
                    embed_dim, hid_dim, feat_dim, 
                    r_type, subgraph,
                    khop, epochs, pt_path):
        print(Fore.RED,Back.YELLOW)
        print('Bigraph network construction!')
        print(Style.RESET_ALL)
        self.nxgraph = self.build_graph(efile, tfile1, tfile2, tfile3, r_type)
        egraph, tgraph = subgraph
        self.node_list = egraph.node_list
        tgraph.node_list = {k+egraph.node_num :v for k, v in tgraph.node_list.items()}
        self.node_list.update(tgraph.node_list)
        with open(file,'r') as f:
            self.elec2road = json.load(f)
        '''
        node list : {node index:  node id}
        0     -- 10886:     elec node
        10887 -- 15711:     road node
        '''
        try:
            feat = {}
            feat['power'] = torch.load(pt_path[0])
            feat['junc'] = torch.load(pt_path[1])
            print('Bigraph features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim,
                                            subgraph,
                                            khop, epochs,
                                            pt_path)
    
    def build_graph(self, efile, tfile1, tfile2, tfile3, r_type):

        print('building bigraph ...')
    
        graph = nx.Graph()
        with open(tfile1, 'r') as f:
            data = json.load(f)
        with open(tfile2, 'r') as f:
            road_type = json.load(f)
        with open(tfile3, 'r') as f:
            tl_id_road2elec_map = json.load(f)
        for road, junc in data.items():
            if len(junc) == 2 and road_type[road] == r_type:
                graph.add_edge(tl_id_road2elec_map[str(junc[0])], tl_id_road2elec_map[str(junc[1])], id=int(road))

        with open(efile, 'r') as f:
            data = json.load(f)
        for key,facility in data.items():
            for node_id in facility.keys():
                node = facility[node_id]
                for neighbor in node['relation']:
                    if int(node_id)<6e8 and (neighbor<6e8) :
                        graph.add_edge(int(node_id),neighbor)
        for tl_id,value in data['tl'].items():
            if int(tl_id) in list(graph.nodes()):
                for neighbor in value['relation']:
                    graph.add_edge(neighbor, int(tl_id))

        print('bigraph builded.')
        return graph
    
    def build_feat(self, embed_dim, hid_dim, feat_dim, subgraph, k, epochs, pt_path):
        egraph, tgraph = subgraph
        n_power = egraph.node_num
        n_junc  = tgraph.node_num
        power_idx = {v:k for k, v in egraph.node_list.items()}
        junc_idx  = {v:k-n_power for k, v in tgraph.node_list.items()}

        edge_list = self.nxgraph.edges()
        elec_edge = [(u, v) for (u, v) in edge_list if u < 6e8 and v < 6e8]
        tran_edge = [(u, v) for (u, v) in edge_list if u > 6e8 and v > 6e8]
        supp_edge = [(u, v) for (u, v) in edge_list 
                                if (u, v) not in elec_edge and (u, v) not in tran_edge]

        elec_src, elec_dst = np.array([power_idx[u] for (u, _) in elec_edge]), np.array([power_idx[v] for (_, v) in elec_edge])
        tran_src, tran_dst = np.array([junc_idx[u] for (u, _) in tran_edge]), np.array([junc_idx[v] for (_, v) in tran_edge])
        supp_src, supp_dst = np.array([junc_idx[u] for (u, _) in supp_edge]), np.array([power_idx[v] for (_, v) in supp_edge])
        
        hetero_graph = dgl.heterograph({
                    ('power', 'elec', 'power'): (elec_src, elec_dst),
                    ('power', 'eleced-by', 'power'): (elec_dst, elec_src),
                    ('junc', 'tran', 'junc'): (tran_src, tran_dst),
                    ('junc', 'traned-by', 'junc'): (tran_dst, tran_src),
                    ('junc', 'supp', 'power'): (supp_src, supp_dst),
                    ('power', 'suppd-by', 'junc'): (supp_dst, supp_src)
                    })

        hetero_graph.nodes['power'].data['feature'] = torch.nn.Embedding(n_power, embed_dim, max_norm=1).weight
        hetero_graph.nodes['junc'].data['feature'] = torch.nn.Embedding(n_junc, embed_dim, max_norm=1).weight
        
        hgcn = HeteroGCN(embed_dim, hid_dim, feat_dim, hetero_graph.etypes)
        
        bifeatures = {
                'junc' :hetero_graph.nodes['junc'].data['feature'],
                'power':hetero_graph.nodes['power'].data['feature']
        }

        optimizer = torch.optim.Adam(hgcn.parameters())
        print('training features ...')
        for epoch in range(epochs):

            num = epoch % 6
            etype = numbers_to_etypes(num)

            t = time.time()
            negative_graph = construct_negative_graph_with_type(hetero_graph, k, etype)
            pos_score, neg_score = hgcn(hetero_graph, negative_graph, bifeatures, etype)
            loss = compute_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                                    " time=", "{:.4f}s".format(time.time() - t)
                                    )

        feat = hgcn.layer(hetero_graph, bifeatures)
        try:
            torch.save(feat['power'], pt_path[0])
            torch.save(feat['junc'], pt_path[1])
            print("saving features sucess")
        except:
            print("saving features failed")

        return feat


BASE = 100000000
geod = Geod(ellps="WGS84")

class ElecNoStep:
    def __init__(self, config, topology, power_10kv, power_load):
        self.basic_config = config["basic_config"]
        self.index_config = config["index_config"]
        self.topology = topology
        self.power_10kv_valid = power_10kv
        self.power_load = power_load
        self.get_fixed_info()
        self.reset()

    def get_fixed_info(self):
        # 状态集合,i,级别，电源，500，220，110，10对应着12345 valid 正常， ruined 直接打击 cascaded 级联 stop 上游没电导致的停电
        self.facility_state_dict_valid = {
            i: {"valid": set(), "ruined": set(), "cascaded": set(),"stopped": set()} for i in range(1, 6)
        }
        for i, key in enumerate(self.topology):
            if i == 4:
                break
            self.facility_state_dict_valid[i+1]["valid"] = set(self.topology[key].keys())
        self.facility_state_dict_valid[5]['valid'] = set([5*BASE + i for i in range(10227)])

        # 计算节点数量
        self.power_num = len(self.topology['power'])
        self.trans500_num = len(self.topology['500kv'])
        self.trans220_num = len(self.topology['220kv'])
        self.trans110_num = len(self.topology['110kv'])
        
        # relation:distance
        self.distance = {}
        key_list = [None, 'power', '500kv', '220kv', '110kv']
        for key in self.topology:
            if key == '110kv':
                break
            for node_id1 in self.topology[key]:
                for node_id2 in self.topology[key][node_id1]["relation"]:
                    if node_id1 < node_id2:
                        key2 = key_list[node_id2 // BASE]
                        # print(node_id1)
                        line_string = LineString(
                            [
                                Point(
                                    self.topology[key][node_id1]['pos'][1], 
                                    self.topology[key][node_id1]['pos'][0]
                                ), 
                                Point(
                                    self.topology[key2][node_id2]['pos'][1], 
                                    self.topology[key2][node_id2]['pos'][0]
                                )
                            ]
                        )
                        self.distance[(node_id1, node_id2)] = geod.geometry_length(line_string) / 1e4
        # self.distance = {}
        # # key_list = [None, 'power', '500kv', '220kv', '110kv']
        # with open('./datasets/elec_flow_input.json', 'r') as f:
        #     data_elec = json.load(f)
        # for geo_data in data_elec:
        #     if geo_data['type'] == 'Feature':
        #         if geo_data["geometry"]["type"] == "LineString":
        #             relation = tuple(geo_data["properties"]["relation"])
        #             if 5 in np.array(relation) // BASE:
        #                 # 去掉10kv的边连接
        #                 continue
        #             self.distance[relation] = None

        # 110kv及以上正常功率
        self.power_110kv_up_valid = {}
        # 创建110kv上下游集合
        self.relation_data_110kv = {}

        

        for key in self.topology['110kv']:
            self.relation_data_110kv[key] = {'relaiton_10':set(), 'relation_220':set()}
            self.power_110kv_up_valid[key] = self.power_load[key] * 1e6
            
            power_key = 0
            for node in self.topology['110kv'][key]['relation']:
                if node // BASE == 5:
                    self.relation_data_110kv[key]['relaiton_10'].add(node)
                    power_key += self.power_10kv_valid[node]
                else:
                    self.relation_data_110kv[key]['relation_220'].add(node)
            

        # for geo_data in data_elec:
        #     if geo_data['type'] == 'Feature':
        #         if geo_data["geometry"]["type"] == "LineString":
        #             relation = tuple(geo_data["properties"]["relation"])
        #             if 4 in np.array(relation) // BASE:
        #                 if 5 in np.array(relation) // BASE:
        #                     node = relation[1]
        #                     self.relation_data_110kv[relation[0]]['relaiton_10'].add(node)
        #                 else:
        #                     node = relation[0]
        #                     self.relation_data_110kv[relation[1]]['relation_220'].add(node)

        # 正常潮流计算矩阵
        self.bus_mat_valid, self.gene_mat_valid, self.branch_mat_valid = self.get_flow_mat()
        
        
        self.bus_mat = self.bus_mat_valid.copy()
        self.branch_mat = self.branch_mat_valid.copy()
        self.gene_mat = self.gene_mat_valid.copy()

        # 220kv以上各点功率
        self.power_110kv_up_valid.update(self.flow_calculate(init=True))

    # init=True时，重置，否则恢复到上一个状态
    def reset(self, init=True):
        if init:
            self.facility_state_dict = copy.deepcopy(self.facility_state_dict_valid)
            self.bus_mat = self.bus_mat_valid.copy()
            self.branch_mat = self.branch_mat_valid.copy()
            self.gene_mat = self.gene_mat_valid.copy()
            self.power_10kv = self.power_10kv_valid.copy()
            self.power_110kv_up = self.power_110kv_up_valid.copy()
        else:
            self.facility_state_dict = copy.deepcopy(self.facility_state_dict_record)
            self.bus_mat = self.bus_mat_record.copy()
            self.branch_mat = self.branch_mat_record.copy()
            self.gene_mat = self.gene_mat_record.copy()
            self.power_10kv = self.power_10kv_record.copy()
            self.power_110kv_up = self.power_110kv_up_record.copy()
    
    def record(self):
        self.facility_state_dict_record = copy.deepcopy(self.facility_state_dict)
        self.bus_mat_record = self.bus_mat.copy()
        self.branch_mat_record = self.branch_mat.copy()
        self.gene_mat_record = self.gene_mat.copy()
        self.power_10kv_record = self.power_10kv.copy()
        self.power_110kv_up_record = self.power_110kv_up.copy()

    def ruin(self, destory_list, flag = 1):
        self.record()
        use_flow = []
        # print(destory_list)
        # print(destory_list[0] // BASE)
        # print(len(self.facility_state_dict[5]["valid"]))
        for id in destory_list:
            if id in self.facility_state_dict[id // BASE]["valid"]:
                self.facility_state_dict[id // BASE]["ruined"].add(id)
                self.facility_state_dict[id // BASE]["valid"].remove(id)
                if id // BASE == 5:
                    self.power_10kv[id] = 0
                elif id // BASE == 4:
                    self.power_110kv_up[id] = 0
                    self.facility_state_dict[5]["stopped"] |= self.relation_data_110kv[id]['relaiton_10']
                    self.facility_state_dict[5]["valid"] -= self.relation_data_110kv[id]['relaiton_10']
                    for id_10 in self.relation_data_110kv[id]['relaiton_10']:
                        self.power_10kv[id_10] = 0
                else:
                    self.power_110kv_up[id] = 0
                    use_flow.append(id)

        if len(use_flow) > 0:
            power_up_220kv = self.flow_calculate(use_flow)
            for key in power_up_220kv:
                if power_up_220kv[key] == 0 and key in self.facility_state_dict[key // BASE]["valid"]:
                    self.facility_state_dict[key // BASE]["cascaded"].add(key)
                    self.facility_state_dict[key // BASE]["valid"].remove(key)
            self.power_110kv_up.update(power_up_220kv)
            
            invalid_220kv = self.facility_state_dict[3]["ruined"] | self.facility_state_dict[3]["cascaded"]
            for id_110 in self.relation_data_110kv:
                if id_110 in self.facility_state_dict[4]["valid"] and self.relation_data_110kv[id_110]['relation_220'] <= invalid_220kv:
                    self.facility_state_dict[4]["stopped"].add(id_110)
                    self.facility_state_dict[4]["valid"].remove(id_110)
                    self.power_110kv_up[id_110] = 0
                    self.facility_state_dict[5]["stopped"] |= self.relation_data_110kv[id_110]['relaiton_10']
                    self.facility_state_dict[5]["valid"] -= self.relation_data_110kv[id_110]['relaiton_10']
                    for id_10 in self.relation_data_110kv[id_110]['relaiton_10']:
                        self.power_10kv[id_10] = 0
        if flag:
            return sum(self.power_10kv.values())
        else:
            return sum(self.power_10kv.values()), self.facility_state_dict

    def flow_calculate(self, use_flow = [], init = False):
        """
        潮流计算，返回损坏的220kv以上节点和220kv以上节点功率
        """
        # 级联循环
        count = 0
        flag = 1
        destory_first = True
        destory_power = []
        destory_220kv = []
        destory_110kv = []
        for key in use_flow:
            if key // BASE == 4:
                destory_110kv.append(key)
            elif key // BASE == 3:
                destory_220kv.append(key)
            else:
                destory_power.append(key)
        while flag:
            if destory_first:
                self.delete_power(destory_power)
                flag_power = np.sum(self.gene_mat[:, self.index_config['GEN_STATUS']])
                if flag_power < 0.1:
                    break
                self.delete_220kv(destory_220kv)
                self.delete_110kv(destory_110kv)
                destory_first = False
            else:
                if cascade_power.size:
                    count += 1
                self.delete_power(cascade_power)
                flag_power = np.sum(self.gene_mat[:, self.index_config['GEN_STATUS']])
                if flag_power < 0.1:
                    break
                self.delete_220kv(cascade_220kv)
                self.delete_110kv()

            ppc = {
                "version": '2',
                "baseMVA": 100.0,
                "bus": self.bus_mat.copy(),
                "gen": self.gene_mat.copy(),
                "branch": self.branch_mat.copy()
            }
            ppopt = ppoption(OUT_GEN=0)
            result, _ = runpf(ppc, ppopt, fname='test')

            if init:
                # 保存power信息，id，正常功率，运行功率，比值
                self.info_gene = np.zeros((self.trans500_num+self.power_num, 4))
                # self.info_gene[:,[0, 1]] = result["gen"][:,[0,1]]
                self.info_gene[:, [0, 1]] = result["branch"][0:25][:, [0, 13]]
                # 保存220kv信息，id，正常功率，运行功率，比值
                self.info_220kv = np.zeros((self.trans220_num, 4))
                index_220kv = np.where(result['branch'][:,self.index_config["TAP"]]==2)[0]
                self.info_220kv[:,[0, 1]] = result["branch"][index_220kv][:,[0,13]]
                break
            
            index_220kv = np.where(result['branch'][:,self.index_config["TAP"]]==2)[0]
            self.info_220kv[:,2] = result["branch"][index_220kv][:,self.index_config["PF"]]
            self.info_220kv[:,3] = abs(self.info_220kv[:,2] / self.info_220kv[:,1])
            self.info_gene[:,2] = result["branch"][0:25][:,self.index_config["PF"]]
            self.info_gene[:,3] = abs(self.info_gene[:,2] / self.info_gene[:,1])
            # print('info_220kv', self.info_220kv)
            cascade_220kv = np.where(
                (self.info_220kv[:,3] >= self.basic_config['up_220'])
            )[0]
            cascade_power = np.where(
                (self.info_gene[:,3] >= self.basic_config['up_power'])
            )[0]
            # print(cascade_power)
            # print(self.info_gene[cascade_power])
            # print(result["gen"][cascade_power])
            flag = len(cascade_220kv) + len(cascade_power)

        update_power_dict = {}
        
        if flag_power < 0.1:
            for i in range(self.power_num):
                id = i + BASE
                update_power_dict[id] = 0
            for i in range(self.trans500_num):
                id = i + BASE * 2
                update_power_dict[id] = 0
            for i in range(self.trans220_num):
                id = i + BASE * 3
                update_power_dict[id] = 0
        else:
            # 更新destory_power中的220kv和500kv功率, 保存220kv及以上{id:功率}到update_power_dict
            update_trans500_power = result["gen"][0:self.trans500_num, self.index_config["PG"]]
            update_gene_power = result["gen"][self.trans500_num:self.trans500_num+self.power_num, self.index_config["PG"]]
            index_220kv = np.where(result['branch'][:,self.index_config["TAP"]]==2)[0]
            update_trans220_power = result["branch"][index_220kv][:, self.index_config["PF"]]

            for i in range(len(update_gene_power)):
                id = i + BASE
                update_power_dict[id] = update_gene_power[i] * 1e6
            for i in range(len(update_trans500_power)):
                id = i + BASE * 2
                update_power_dict[id] = update_trans500_power[i] * 1e6
            for i in range(len(update_trans220_power)):
                id = i + BASE * 3
                update_power_dict[id] = update_trans220_power[i] * 1e6

        return update_power_dict

    def delete_power(self, destory_power_list):
        # 处理电源
        for id in destory_power_list:
            if id // BASE == 2:
                self.gene_mat[id % BASE, self.index_config['GEN_STATUS']] = 0
                # self.branch_mat[id % BASE, self.index_config['BR_STATUS']] = 0
            elif id // BASE == 1:
                self.gene_mat[id % BASE + self.trans500_num, self.index_config['GEN_STATUS']] = 0
                # self.branch_mat[id % BASE + self.trans500_num, self.index_config['BR_STATUS']] = 0
                # condi = np.where(
                #     self.bus_mat[:, self.index_config['BUS_I']] == id % BASE + self.trans500_num
                # )[0]
                # self.bus_mat = np.delete(self.bus_mat, condi, axis=0)
                # condi = np.where(
                #     self.branch_mat[:, self.index_config['F_BUS']] == id % BASE + self.trans500_num
                # )[0]
                # self.branch_mat = np.delete(self.branch_mat, condi, axis=0)
    def delete_220kv(self, destory_220kv_list):
        # 处理220kv
        for id in destory_220kv_list:
            connect_110kv = np.where(
                self.branch_mat[:,0] == id % BASE + 2 * (self.power_num + self.trans500_num) + self.trans220_num
            )[0]
            self.branch_mat = np.delete(self.branch_mat, connect_110kv, axis=0)

    def delete_110kv(self, destory_110kv = []):
        # 直接摧毁, 此时110kv branch，bus均有, 潮流计算矩阵在不断删除，智能用np.where查找
        if destory_110kv:
            for id in destory_110kv:
                bus_id = id % BASE + 2 * (self.trans500_num + self.power_num + self.trans220_num)
                condi_110kv = np.where(
                    self.branch_mat[:, 1] == bus_id
                )[0]
                self.branch_mat = np.delete(self.branch_mat, condi_110kv, axis=0)
                bus_110kv = np.where(
                    self.bus_mat[:,0] == bus_id
                )[0]
                self.bus_mat = np.delete(self.bus_mat, bus_110kv, axis=0)
        # 110kv的上游两个220kv如果被删除，删除110kv
        else:
            for id in range(self.trans110_num):#110kv数量
                bus_id = id % BASE + 2 * (self.trans500_num + self.power_num + self.trans220_num)
                condi_110kv = np.where(
                    self.branch_mat[:, 1] == bus_id
                )[0]
                if condi_110kv.size == 0:
                    # 删除bus
                    bus_110kv = np.where(
                        self.bus_mat[:,0] == bus_id
                    )[0]
                    if len(bus_110kv) != 0:#有可能已经删了
                        self.bus_mat = np.delete(self.bus_mat, bus_110kv, axis=0)
    
    def get_flow_mat(self):
        """
        由self.topology获取正常运行的flow_mat，每个拓扑只运行一次
        参看pypower文档
        """
        Bus_data = []
        Generator_data = []
        Branch_data = []

        # Bus data Generate
        Bus_num = 2 * (self.trans500_num + self.power_num + self.trans220_num) + self.trans110_num 
        for bus_id in range(Bus_num):
            type_id = 1
            Pd = 0
            Qd = 0
            # 与500kv连接的母线
            if bus_id < self.trans500_num:
                type_id = 3
                Vm = 5/1.1
            # 与发电厂连接的母线
            elif bus_id < self.trans500_num + self.power_num:
                type_id = 3
                Vm = 5/1.1
            # 500kv、发电厂出线以及220kv入线
            elif bus_id < 2 * (self.trans500_num + self.power_num) + self.trans220_num:
                Vm = 2
            # 220kv出线
            elif bus_id < 2 * (self.trans500_num + self.power_num + self.trans220_num):
                Vm = 1
            # 110kv入线
            else:
                index_110 = int(bus_id - 2 * (self.trans500_num + self.power_num + self.trans220_num)) + 4 * BASE
                Pd = self.power_110kv_up_valid[index_110] / 1e6
                Qd = Pd * math.sqrt((1 / self.basic_config["cos_phi"]) ** 2 - 1)
                Vm = 1
            Bus_data.append(
                [
                    bus_id,
                    type_id,
                    Pd,
                    Qd,
                    self.basic_config["Gs"],
                    self.basic_config["Bs"],
                    self.basic_config["area"],
                    Vm,
                    self.basic_config["Va"],
                    self.basic_config["baseKV"],
                    self.basic_config["Zone"],
                    Vm * 1.5,   #Vmax
                    Vm / 1.5    #Vmin
                ]
            )

        # Generator data generate
        Generator_num = self.trans500_num + self.power_num
        for bus_id in range(Generator_num):
            # 500kv电源
            if bus_id < self.trans500_num:
                Pg = 0
                Qg = 0
            # 发电厂
            else:
                Pg = 0
                Qg = 0
            Generator_data.append(
                [
                    bus_id,
                    Pg,
                    Qg,
                    self.basic_config["Qmax"],
                    self.basic_config["Qmin"],
                    self.basic_config["Vg"],
                    self.basic_config["mbase"],
                    self.basic_config["status"],
                    self.basic_config["Pmax"],
                    self.basic_config["Pmin"],
                    self.basic_config["Pc1"],
                    self.basic_config["Pc2"],
                    self.basic_config["Qc1min"],
                    self.basic_config["Qc1max"],
                    self.basic_config["Qc2min"],
                    self.basic_config["Qc2max"],
                    self.basic_config["ramp_agc"],
                    self.basic_config["ramp_10"],
                    self.basic_config["ramp_30"],
                    self.basic_config["ramp_q"],
                    self.basic_config["apf"],
                ]
            )

        # 变压器支路
        Transformer_num = self.trans500_num + self.power_num + self.trans220_num
        for i in range(Transformer_num):
            r = 0
            x = 1e-6
            b = 0
            if i < self.trans500_num + self.power_num:
                fbus = i
                tbus = fbus + self.trans500_num + self.power_num
                ratio = 5/2.2
            else:
                fbus = i + self.trans500_num + self.power_num
                tbus = fbus + self.trans220_num
                ratio = 2
            Branch_data.append(
                [
                    fbus,
                    tbus,
                    r,
                    x,
                    b,
                    self.basic_config["rateA"],
                    self.basic_config["rateB"],
                    self.basic_config["rateC"],
                    ratio,
                    self.basic_config["angle"],
                    self.basic_config["status"],
                    self.basic_config["angmin"],
                    self.basic_config["angmax"],
                ]
            )

        for relation in self.distance:
            if relation[1] // BASE == 4:
                fbus = relation[0] % BASE + 2 * (self.trans500_num + self.power_num) + self.trans220_num
                tbus = relation[1] % BASE + 2 * (self.trans500_num + self.power_num + self.trans220_num)
            elif relation[1] // BASE == 3:
                tbus = relation[1] % BASE + 2 * (self.trans500_num + self.power_num)
                if relation[0] // BASE == 3:
                    fbus = relation[0] % BASE + 2 * (self.trans500_num + self.power_num)
                elif relation[0] // BASE == 2:
                    fbus = relation[0] % BASE + self.trans500_num + self.power_num
                else:
                    fbus = relation[0] % BASE + 2 * (self.trans500_num) + self.power_num
            else:
                fbus = relation[0] % BASE
                tbus = relation[1] % BASE
            # distance = self.distance[relation]
            # r = 0.000023 * distance
            # x = 0.000031 * distance
            # b = 3.765 * distance * 1e-8
            r = 1e-3
            x = 1e-3
            b = 1e-8
            ratio = 1
            Branch_data.append(
                [
                    fbus,
                    tbus,
                    r,
                    x,
                    b,
                    self.basic_config["rateA"],
                    self.basic_config["rateB"],
                    self.basic_config["rateC"],
                    ratio,
                    self.basic_config["angle"],
                    self.basic_config["status"],
                    self.basic_config["angmin"],
                    self.basic_config["angmax"],
                ]
            )

        return np.array(Bus_data), np.array(Generator_data), np.array(Branch_data)


class Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.hidden = nn.Linear(in_dim, hid_dim)
        self.output = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        hidden = self.hidden(inputs)
        outputs = self.relu(hidden)
        outputs = self.output(outputs)

        return outputs

class DQN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, 
                memory_capacity, iter, batch_size,
                lr, epsilon, gamma,
                label, model_pt):
        super().__init__()

        self.embed_dim = in_dim
        if label == 'train':
            self.enet = Net(in_dim, hid_dim, out_dim).to(device)
        elif label == 'test':
            self.enet = Net(in_dim, hid_dim, out_dim).to(device)
            self.enet.load_state_dict(torch.load(model_pt))
        else:
            pass
        
        self.tnet = Net(in_dim, hid_dim, out_dim).to(device)
        self.learning_step = 0
        self.memory_num = 0
        self.mem_cap = memory_capacity
        self.iter = iter
        self.bsize = batch_size
        self.reply_buffer = np.zeros((
                            memory_capacity,
                            in_dim * 2 + 2
                            ))

        self.optimizer = torch.optim.Adam(self.enet.parameters(), lr)
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon

    def choose_node(self, features, state, choosen, exist):

        node_num = features.shape[0]
        if np.random.uniform() < self.epsilon:
            outputs = self.enet(features).to(device)
            s_mat = torch.tile(state, (node_num, 1)).to(device)
            Q_values = torch.sum(outputs * s_mat, axis=1).reshape(node_num, -1).to(device)
            Q_cp = Q_values.data.cpu().numpy()
            Q_cp[choosen] = -1e8
            node = int(np.argmax(Q_cp)) 

        else:
            node = random.sample(exist, 1)[0]
        
        return node

    def attack(self, features, state, choosen):

        node_num = features.shape[0]
        outputs = self.enet(features).to(device)
        s_mat = torch.tile(state, (node_num, 1)).to(device)
        Q_values = torch.sum(outputs * s_mat, axis=1).reshape(node_num, -1).to(device)
        Q_cp = Q_values.data.cpu().numpy()
        Q_cp[choosen] = -1e8
        node = int(np.argmax(Q_cp)) 

        return node


    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, [a, r], s_)) 

        # replace the old memory with new memory
        index = self.memory_num % self.mem_cap
        self.reply_buffer[index, :] = transition
        self.memory_num += 1

    def learn(self, features):

        node_num = features.shape[0]
        # target net parameters update
        if self.learning_step % self.iter == 0:       
            self.tnet.load_state_dict(self.enet.state_dict())
        self.learning_step += 1

        # sample batch transitions
        sample_idx = np.random.choice(self.mem_cap, self.bsize)
        b_memory = self.reply_buffer[sample_idx, :]
        b_s = b_memory[:, :self.embed_dim]
        b_a = torch.LongTensor(b_memory[:, self.embed_dim: self.embed_dim + 1].astype(int)).reshape(self.bsize, -1).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.embed_dim+1:self.embed_dim+2]).to(device)
        b_s_ = b_memory[:, -self.embed_dim:]

        # q_eval
        eval_out = self.enet(features).to(device)
        q_eval = torch.zeros((self.bsize, node_num), dtype=torch.float).to(device)
        for idx, s in enumerate(b_s):
            s_mat = torch.FloatTensor(np.tile(s, (node_num, 1))).to(device)
            Q = torch.sum((eval_out * s_mat), axis=1).reshape(node_num, -1).to(device)
            q_eval[idx] = Q.reshape(-1, node_num)
        
        # q_targ
        targ_out = self.tnet(features).to(device)
        q_next = torch.zeros((self.bsize, node_num), dtype=torch.float).to(device)
        for idx, _s in enumerate(b_s_):
            _s_mat = torch.FloatTensor(np.tile(_s, (node_num, 1))).to(device)
            Q = torch.sum((targ_out * _s_mat), axis=1).reshape(node_num, -1).to(device)
            q_next[idx] = Q.reshape(-1, node_num)

        # print(q_eval.shape)   [20, 10887]
        # print(q_next.shape)   [20, 10887]

        # q_target
        q_eval = torch.gather(q_eval, 1, b_a).to(device)
        q_target = b_r + (self.gamma * torch.max(q_next,dim=1)[0]).reshape(self.bsize, -1).to(device)

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return

class Regressor(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Regressor, self).__init__()
        self.hidden = nn.Linear(input_dim, hid_dim)
        self.output = nn.Linear(hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, feat):
        out = self.hidden(feat)
        out = self.relu(out)
        out = self.output(out)

        return self.sigmoid(out)
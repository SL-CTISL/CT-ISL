import numpy as np
import sklearn
from os.path import join as pjoin
import os
import networkx as nx
from tqdm import tqdm


class DataReader_snapshot():
    def __init__(self,
                 rumor_cascade,  
                 rnd_state=None,
                 folds=10,
                 union_graph_inf = None):

        self.data = {}
        self.rnd_state = rnd_state
        self.rumor_extend_cascade = rumor_cascade
        self.union_graph_inf = union_graph_inf

        self.graph_nodes = [len(g.nodes) for g in rumor_cascade]
        self.max_nodes = max(self.graph_nodes)
        self.graph_num = len(self.graph_nodes)
      
        permutation = np.arange(self.graph_num)
        train_ids, test_ids = split_ids(permutation, folds=folds)

        
        splits = []
        for fold in range(
                len(train_ids)):  
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})
        self.data['splits'] = splits
        self.generate_data()

        print()
        print()

    def generate_data(self):
        adj_list = []
        targets = []
        snapshots = []
        features_Ts = []
        influence = []

        for g in tqdm(self.rumor_extend_cascade):
            
            uid_to_index = {uid: idx for idx, uid in enumerate(g.nodes)}

            
            adj_matrix = np.zeros((len(g.nodes), len(g.nodes)))
            for u, v in g.edges():
                if u == v:
                    continue
                adj_matrix[uid_to_index[u], uid_to_index[v]] = 1
                adj_matrix[uid_to_index[v], uid_to_index[u]] = 1
            adj_list.append(adj_matrix)

            
            times = nx.get_node_attributes(g, 'time')
            source_uid = min(times, key=times.get)  
            source_idx = uid_to_index[source_uid]

            
            one_hot_target = np.zeros(len(g.nodes))
            one_hot_target[source_idx] = 1
            targets.append(one_hot_target)

            snapshot_for_g = np.zeros(len(g.nodes))
            influence_for_g = []

            infected_nodes = sorted([node for node in g.nodes if g.nodes[node]['time'] < 999999],
                                    key=lambda x: g.nodes[x]['time'])
            num_infected_nodes_to_select = max(1, int(0.1 * len(infected_nodes)))  
            
            selected_infected_nodes = infected_nodes[:num_infected_nodes_to_select]
            
            for node in selected_infected_nodes:
                index = uid_to_index[node]
                snapshot_for_g[index] = 1

            
            
            node_info = {}
            for node in g.nodes():
                neighbors = list(g.neighbors(node))
                infected_neighbors = [n for n in neighbors if g.nodes[n]['time'] <= g.nodes[infected_nodes[num_infected_nodes_to_select]]['time']]
                uninfected_neighbors = [n for n in neighbors if g.nodes[n]['time'] > g.nodes[infected_nodes[num_infected_nodes_to_select]]['time']]
                node_info[node] = {
                    'neighbors': neighbors,
                    'infected_neighbors': infected_neighbors,
                    'uninfected_neighbors': uninfected_neighbors,
                    'infected_ratio': len(infected_neighbors) / len(neighbors) if neighbors else 0,
                    'time': g.nodes[node]['time']
                }
            max_infected_neighbors = max([len(info['neighbors']) for info in node_info.values()])

            def get_node_features_for_snapshotv2(node, g, union_graph_inf, node_info, max_infected_neighbors, last_infected_time, random_times=None, max_time=None):
                
                features = list(union_graph_inf.nodes[node]['norm_profile_list']) 

                
                features.append(len(node_info[node]['infected_neighbors']) / max_infected_neighbors) 
                features.append(len(node_info[node]['uninfected_neighbors']) / max_infected_neighbors) 
                features.append(node_info[node]['infected_ratio']) 
                features.append(1 - node_info[node]['infected_ratio']) 

                features.append(len(node_info[node]['neighbors']) / max_infected_neighbors)  
                assert len(node_info[node]['neighbors']) == len(node_info[node]['infected_neighbors']) + len(node_info[node]['uninfected_neighbors']), 'wrong!'

                
                is_infected = 1 if g.nodes[node]['time'] <= last_infected_time else 0
                features.append(is_infected) 
                features.append(1 - is_infected) 

                return features

            features_Ts_for_g = []
            for uid in uid_to_index:
                
                features = get_node_features_for_snapshotv2(uid, g, self.union_graph_inf, node_info,
                                                          max_infected_neighbors, g.nodes[infected_nodes[num_infected_nodes_to_select]]['time'], random_times=None, max_time=None)
                
                features_Ts_for_g.append(features)
                influence_for_g.append(self.union_graph_inf.nodes[uid]['inf'])

            features_Ts.append(features_Ts_for_g)
            snapshots.append(snapshot_for_g)
            influence.append(influence_for_g)

        self.data['adj_list'] = np.array(adj_list)
        self.data['targets'] = np.array(targets)
        self.data['snapshots'] = np.array(snapshots)
        self.data['features_Ts'] = np.array(features_Ts)
        self.data['influence'] = np.array(influence)


def split_ids(ids, folds=10):  
    
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))  
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    train_ids = []
    for fold in range(folds):
        train_ids.append(np.array(
            [e for e in ids if e not in test_ids[fold]]))  
        assert len(train_ids[fold]) + len(test_ids[fold]) == len(
            np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids
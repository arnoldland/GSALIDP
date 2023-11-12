import os.path as osp
import numpy as np
import os
import torch
from torch_geometric.data import Dataset
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

class SiteDataset(Dataset):
    def __init__(
            self, 
            root, 
            transform=None, 
            pre_transform=None
        ):
        super(SiteDataset, self).__init__(root, transform, pre_transform)
        self.root = root

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        raw_list=['./contact_graph', './raw_data/monomer', './raw_data/pair_interaction_site']
        return raw_list
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    
    @property
    def processed_file_names(self):
        processed_list=[]
        for i in range(0,34):
            processed_list.append('data'+str(i)+'.pt')
        return processed_list

    def download(self):
        print('Raw data is not prepared yet.')
        pass

    def process(self):
        for k in range(1,35):
            x_t = []
            edge_index_t = []
            edge_weight_t = [] # all 1
            y_t = []
            for m in range(1,17):
                t = round(0.0 - 0.2*(16-m), 1)
                idx_features = np.genfromtxt('./raw_data/monomer/feature/'+str(k)+'_'+str(t)+'ns-A.info')
                x = idx_features[:, 4:15]
                num_node = len(x)
                idx = np.array(idx_features[:, 2], dtype=np.int32)
                idx = idx - 2 # id from 0
                id_node = {j: i for i, j in enumerate(idx)}
                edges_unordered = np.genfromtxt('./contact_graph/contactg_'+str(k)+'_'+str(t)+'ns-A.csv', encoding='utf-8', delimiter=',', dtype=np.int32)
                edge_str = [id_node[each[0]] for each in edges_unordered] # dismap from 0 to 167
                edge_end = [id_node[each[1]] for each in edges_unordered]
                edge_index = np.array([edge_str, edge_end], dtype=np.int32)
                edge_weight = np.ones(len(edge_str))
                x_t.append(x)
                edge_index_t.append(edge_index)
                if m == 1 :    
                    y = self.encode_labels(
                    np.genfromtxt('./raw_data/pair_interaction_site/' + str(k) + '.ctp', encoding='utf-8', dtype=np.int32, usecols=(2)),
                    num_node=num_node) 
                y_t.append(y)
                edge_weight_t.append(edge_weight)
            data = DynamicGraphTemporalSignal(edge_indices=edge_index_t,edge_weights= edge_weight_t, features=x_t, targets=y_t)
            torch.save(data, os.path.join(self.processed_dir, 'data'+str(k-1)+'.pt'))

    def encode_labels(self, interaction, num_node):
        labels = np.zeros(num_node, dtype=np.int32)
        labels[interaction-2] = 1 #origin_id 2 to 169, new id 0 to 167
        return labels

    def len(self):
        return len(self.processed_file_names)

    def get(self, index):
        data = torch.load(os.path.join(self.processed_dir,'data'+str(index)+'.pt'))
        return data

class PairDataset(Dataset):
    def __init__(
            self, 
            root, 
            transform=None, 
            pre_transform=None
        ):
        super(PairDataset, self).__init__(root, transform, pre_transform)
        self.root = root

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        raw_list=['./contact_graph', './raw_data/monomer', './raw_data/pair_interaction_site']
        return raw_list
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    
    @property
    def processed_file_names(self):
        processed_list=[]
        for i in range(0,34):
            processed_list.append('data_a'+str(i)+'.pt')
            processed_list.append('data_b'+str(i)+'.pt')
        return processed_list

    def download(self):
        print('Raw data is not prepared yet.')
        pass

    def process(self):
        for k in range(1,35):
            x_a_t = []
            y_a_t = []
            edge_index_a_t = []
            edge_weights_a_t = []

            x_b_t = []
            y_b_t = []
            edge_index_b_t = []
            edge_weights_b_t = []

            for m in range(1,17):
                t = round(0.0 - 0.2*(16-m), 1)
                idx_features_a = np.genfromtxt('./raw_data/monomer/feature/'+str(k)+'_'+str(t)+'ns-A.info')
                x_a = idx_features_a[:, 4:15]
                num_node_a = len(x_a) 
                idx_a = np.array(idx_features_a[:, 2], dtype=np.int32)
                idx_a = idx_a - 2 #id from 0
                id_node_a = {j: i for i, j in enumerate(idx_a)}
                edges_unordered_a = np.genfromtxt('./contact_graph/contactg_'+str(k)+'_'+str(t)+'ns-A.csv', encoding='utf-8', delimiter=',', dtype=np.int32)
                edge_str_a = [id_node_a[each[0]] for each in edges_unordered_a] #dismap from 0 to 167
                edge_end_a = [id_node_a[each[1]] for each in edges_unordered_a]
                edge_index_a = np.array([edge_str_a, edge_end_a], dtype=np.int32)
                edge_weights_a = np.ones(len(edge_str_a))
                x_a_t.append(x_a)
                edge_index_a_t.append(edge_index_a)
                edge_weights_a_t.append(edge_weights_a)


                idx_features_b = np.genfromtxt('./raw_data/monomer/feature/' + str(k) + '_' + str(t) + 'ns-B.info')
                x_b = idx_features_b[:, 4:15]
                num_node_b = len(x_b)
                idx_b = np.array(idx_features_b[:, 2], dtype=np.int32) 
                idx_b = idx_b - 2 #id from 0
                edges_unordered_b = np.genfromtxt('./contact_graph/contactg_'+str(k) + '_' + str(t) + 'ns-B.csv',
                                                  encoding='utf-8', delimiter=',', dtype=np.int32)
                edge_str_b = [id_node_a[each[0]] for each in edges_unordered_b] #dismap from 0 to 167
                edge_end_b = [id_node_a[each[1]] for each in edges_unordered_b]
                edge_index_b = np.array([edge_str_b, edge_end_b], dtype=np.int32)
                edge_weights_b = np.ones(len(edge_str_b))
                x_b_t.append(x_b)
                edge_index_b_t.append(edge_index_b)
                edge_weights_b_t.append(edge_weights_b)
                if m == 1:
                    interaction = np.genfromtxt('./raw_data/pair_interaction_site/' + str(k) + '.ctp', encoding='utf-8', dtype=np.int32, usecols=(2,4))
                    y_a = self.encode_labels(interaction,num_node=num_node_a,mode=0)  # read 2 and 4
                    y_b = self.encode_labels(interaction,num_node=num_node_b,mode=1)
                y_a_t.append(y_a)
                y_b_t.append(y_b)

            data_a = DynamicGraphTemporalSignal(edge_indices=edge_index_a_t, edge_weights=edge_weights_a_t ,features=x_a_t, targets=y_a_t)
            torch.save(data_a, os.path.join(self.processed_dir, 'data_a'+str(k-1)+'.pt'))
            data_b = DynamicGraphTemporalSignal(edge_indices=edge_index_b_t, edge_weights=edge_weights_b_t, features=x_b_t, targets=y_b_t)
            torch.save(data_b, os.path.join(self.processed_dir, 'data_b'+str(k-1)+'.pt'))

    def encode_labels(self, interaction, num_node, mode):
        labels = np.zeros(num_node, dtype=np.int32)
        if mode == 0:
            for each in interaction:
                labels[each[0]-2] = each[1]-2
        else:
            for each in interaction:
                labels[each[1]-2] = each[0]-2
        return labels

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data_a = torch.load(os.path.join(self.processed_dir,'data_a' +str(idx)+'.pt'))
        data_b = torch.load(os.path.join(self.processed_dir, 'data_b' + str(idx) + '.pt'))
        return data_a, data_b
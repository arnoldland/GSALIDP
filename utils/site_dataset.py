import os.path as osp
import numpy as np
import os
import torch
from torch_geometric.data import Dataset
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

class IDPDataset(Dataset):
    def __init__(
            self, 
            root, 
            transform=None, 
            pre_transform=None
        ):
        super(IDPDataset, self).__init__(root, transform, pre_transform)
        self.root = root
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
        pass

    def process(self):
        for k in range(1,35):
            x_t = []
            edge_index_t = []
            edge_weight_t = [] # all 1
            y_t = []
            for m in range(1,17):
                t = round(0.0 - 0.2*(16-m), 1)
                idx_features = np.genfromtxt('./raw/newinfo/'+str(k)+'_'+str(t)+'ns-A.info')
                x = idx_features[:,[4:15]]
                num_node = len(x)
                idx = np.array(idx_features[:, 2], dtype=np.int32)
                idx = idx - 2 # id from 0
                id_node = {j: i for i, j in enumerate(idx)}
                edges_unordered = np.genfromtxt('./raw/graph_d/compactg_'+str(k)+'_'+str(t)+'ns-A.csv', encoding='utf-8', delimiter=',', dtype=np.int32)
                edge_str = [id_node[each[0]] for each in edges_unordered] # dismap from 0 to 167
                edge_end = [id_node[each[1]] for each in edges_unordered]
                edge_index = np.array([edge_str, edge_end], dtype=np.int32)
                edge_weight = np.ones(len(edge_str))
                x_t.append(x)
                edge_index_t.append(edge_index)
                if m == 1 :    
                    y = self.encode_labels(
                    np.genfromtxt('./raw/pair/' + str(k) + '.ctp', encoding='utf-8', dtype=np.int32, usecols=(2)),
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

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir,'data'+str(idx)+'.pt'))
        return data

import torch
import copy
import numpy as np

class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 datareader,
                 fold_id,
                 split):
        self.fold_id = fold_id
        self.split = split
        self.rnd_state = datareader.rnd_state
        self.set_fold(datareader.data, fold_id)

    def set_fold(self, data, fold_id):
        self.total = len(data['targets'])
        self.idx = data['splits'][fold_id][self.split]      
        self.snapshots = copy.deepcopy([data['snapshots'][i] for i in self.idx])  
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])  
        
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.features_Ts = copy.deepcopy([data['features_Ts'][i] for i in self.idx])  
        self.influence = copy.deepcopy([data['influence'][i] for i in self.idx])

        print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))

    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, index):
        
        
        return [
                torch.from_numpy(self.adj_list[index]).float(),  
                torch.from_numpy(self.labels[index]),            
                torch.from_numpy(np.stack(self.snapshots[index])),         
                torch.from_numpy(np.stack(self.features_Ts[index])).float(),  
                torch.from_numpy(np.stack(self.influence[index])).float(),
        ]
        
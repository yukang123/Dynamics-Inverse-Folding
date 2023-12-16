import torch
import numpy as np
from ema_pytorch import EMA
from gradeif import EGNN_NET,GraDe_IF
import torch.nn.functional as F

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
                
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
ckpt = torch.load('results/weight/UNIFORM_3M_small.pt', map_location=device)#BLOSUM_3M_small.pt
config = ckpt['config']
config['noise_type'] = 'uniform'#blosum


gnn = EGNN_NET(
    input_feat_dim=config['input_feat_dim'],hidden_channels=config['hidden_dim'],
    edge_attr_dim=config['edge_attr_dim'],dropout=config['drop_out'],
    n_layers=config['depth'],
    update_edge = config['update_edge'],
    embedding=config['embedding'],embedding_dim=config['embedding_dim'],
    embed_ss=config['embed_ss'],norm_feat=config['norm_feat'])

diffusion = GraDe_IF(model = gnn,config=config)

diffusion = EMA(diffusion)
diffusion.load_state_dict(ckpt['ema'])

import sys
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import torch_geometric
from torch_geometric.data import Batch
from dataset_src.generate_graph import prepare_graph,pdb2graph

pdb_path = '../dataset/raw/test/3fkf.A.pdb'
graph = pdb2graph(pdb_path,normalize_path = '../dataset_src/mean_attr.pt')
input_graph = Batch.from_data_list([prepare_graph(graph)])
print(input_graph)

import sys
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import torch_geometric
from torch_geometric.data import Batch
from dataset_src.generate_graph import prepare_graph,pdb2graph

pdb_path = '../dataset/raw/test/3fkf.A.pdb'
graph = pdb2graph(pdb_path,normalize_path = '../dataset_src/mean_attr.pt')
input_graph = Batch.from_data_list([prepare_graph(graph)])
print(input_graph)


import networkx as nx
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,figsize=(5, 5))
g = torch_geometric.utils.to_networkx(input_graph)
nx.draw(g,node_size=0.5,pos=input_graph.pos[:,1:3].cpu().numpy(),ax=ax)

ensemble_num = 50
all_prob = []
for i in range(ensemble_num):
    prob,sample_graph = diffusion.ema_model.ddim_sample(input_graph,step=50,diverse=True)
    all_prob.append(prob)

all_zt_tensor = torch.stack(all_prob)
recovery = (all_zt_tensor.mean(dim = 0).argmax(dim=1) == input_graph.x.argmax(dim = 1)).sum()/input_graph.x.shape[0]
print('recovery rate:', recovery.item())
ll_fullseq = F.cross_entropy(all_zt_tensor.mean(dim = 0),input_graph.x.argmax(dim = 1), reduction='mean').item()
perplexity = np.exp(ll_fullseq)
print('perplexity:' , perplexity)

ll_fullseq = F.cross_entropy(all_prob[0],input_graph.x.argmax(dim = 1), reduction='mean').item()
perplexity = np.exp(ll_fullseq)
print('perplexity:' , perplexity)


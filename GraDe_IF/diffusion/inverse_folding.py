import torch
import numpy as np
from ema_pytorch import EMA
from gradeif import EGNN_NET,GraDe_IF
import torch.nn.functional as F
import sys
print(sys.path)

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
                
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ckpt = torch.load('results/weight/D3PM_UNIFORM_3M.pt', map_location=device)
ckpt = torch.load('results/weight/UNIFORM_3M_small.pt', map_location=device)
print(ckpt.keys())
config = ckpt['config']
config['noise_type'] = 'uniform'
print(config.keys())

gnn = EGNN_NET(
    input_feat_dim=config['input_feat_dim'],hidden_channels=config['hidden_dim'],
    edge_attr_dim=config['edge_attr_dim'],dropout=config['drop_out'],
    n_layers=config['depth'],
    # update_edge = config['update_edge'],
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
# pdb_path = "../../../Data/DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion/sarscov2-15235449-peptide-A-no-water-no-ion/sarscov2-15235449-peptide-A-no-water-no-ion-0000-pdbs/0000.pdb"
graph = pdb2graph(pdb_path,normalize_path = '../dataset_src/mean_attr.pt')
input_graph = Batch.from_data_list([prepare_graph(graph)])
print(input_graph)
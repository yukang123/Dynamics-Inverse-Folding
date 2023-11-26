import sys
sys.path.append('diffusion')

import torch
from torch_geometric.data import Batch
from diffusion.gradeif import GraDe_IF,EGNN_NET
from dataset_src.generate_graph import prepare_graph

gnn = EGNN_NET(input_feat_dim=input_graph.x.shape[1]+input_graph.extra_x.shape[1],hidden_channels=10,edge_attr_dim=input_graph.edge_attr.shape[1])

diffusion_model = GraDe_IF(gnn)

graph = torch.load('dataset/process/test/3fkf.A.pt')
input_graph = Batch.from_data_list([prepare_graph(graph)])

loss = diffusion_model(input_graph)
loss.backward()

_,sample_seq = diffusion_model.ddim_sample(input_graph) #using structure information generate sequence
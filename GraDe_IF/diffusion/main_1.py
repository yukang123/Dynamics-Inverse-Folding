import torch
import numpy as np
from ema_pytorch import EMA
from gradeif import EGNN_NET,GraDe_IF
import torch.nn.functional as F
import sys
import os
# os.environ["DSSP"] = "/scratch/network/yy1325/Dynamics-Inverse-Folding/mkdssp"
import ot 
import torch_geometric
from torch_geometric.data import Batch
from dataset_src.generate_graph import prepare_graph,pdb2graph

import networkx as nx
import matplotlib.pyplot as plt

# current_directory = os.getcwd()
os.chdir(os.path.dirname(__file__))
current_directory = os.path.dirname(__file__)
print(current_directory)
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)




amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
                
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# print(diffusion.device)


def sample_sequence(pdb_path, ali_folder, ignore_chains = ["A"], sample_num=1, step_interval = 50, sta_end=[], batch_size=1):
    assert len(sta_end) == 2 or len(sta_end) == 0

    # pdb_path = '../dataset/raw/test/3fkf.A.pdb'
    graph = pdb2graph(pdb_path,normalize_path = '../dataset_src/mean_attr.pt', ignore_chains=ignore_chains)
    # input_graph = Batch.from_data_list([prepare_graph(graph)])

    input_graph = Batch.from_data_list([prepare_graph(graph)]*batch_size)

    # print(input_graph)
    # fig,ax = plt.subplots(1,figsize=(5, 5))
    # g = torch_geometric.utils.to_networkx(input_graph)
    # nx.draw(g,node_size=0.5,pos=input_graph.pos[:,1:3].cpu().numpy(),ax=ax)

    if len(sta_end) == 2:
        # valid_ind = torch.zeros(input_graph.x.shape[0])
        # valid_ind[sta_end[0], sta_end[1]] = 1
        start, end = sta_end
    else:
        # valid_ind = torch.ones(input_graph.x.shape[0])
        start, end = 0, input_graph.x.shape[0]

    native_seq = ''.join([amino_acids_type[i] for i in input_graph.x.argmax(dim=1).tolist()])[start: end]
    name_ = os.path.splitext(os.path.basename(pdb_path))[0]
    print("Gtruth sequence: ",  native_seq)

    ali_file = os.path.join(ali_folder, "{}.fa".format(name_))
    sample_graphs = []
    sample_probs = []
    recovery_list = []
    with open(ali_file, "w") as f:
        f.write('>{}, ignored_chains={}\n{}\n'.format(
            name_, ignore_chains, native_seq
        ))
        for i in range(sample_num):
            prob,sample_graph = diffusion.ema_model.ddim_sample(input_graph, step=step_interval, diverse=True)
            recovery = (prob.argmax(dim=1) == input_graph.x.argmax(dim = 1))[start:end].sum()/(input_graph.x)[start:end].shape[0]
            sample_seq = ''.join([amino_acids_type[i] for i in sample_graph.argmax(dim=1).tolist()])[start:end]
            # print('sample sequence: ', sample_seq )
            # print('recovery rate:', recovery.item())
            f.write('>Sample={}, seq_recovery={:.4f}\n{}\n'.format(
                i,
                # prob,
                recovery, sample_seq)) #write generated sequence
            sample_graphs.append(sample_graph)
            sample_probs.append(prob[start:end])
            score = F.cross_entropy(prob[start:end],sample_graph.argmax(dim = 1)[start:end], reduction='mean').item()
            recovery_list.append(recovery)
        sample_probs_tensor = torch.cat(sample_probs)
        all_seqs = torch.cat([input_graph.x.argmax(dim=1)[start:end]]*sample_num)

        all_zt_tensor = torch.stack(sample_probs)
        print(all_zt_tensor.mean(dim = 0).shape)
        recovery = (all_zt_tensor.mean(dim = 0).argmax(dim=1) == input_graph.x[start: end].argmax(dim = 1)).sum()/input_graph.x[start:end].shape[0]
        print('recovery rate:', recovery.item())
        ll_fullseq_n = F.cross_entropy(all_zt_tensor.mean(dim = 0),input_graph.x.argmax(dim = 1)[start:end], reduction='mean').item()
        perplexity = np.exp(ll_fullseq_n)
        print("perplexity:", perplexity)
        f.write("accumulated score: {}\n".format(ll_fullseq_n))
        f.write("perplexity: {}\n".format(perplexity))

        ll_fullseq = F.cross_entropy(sample_probs_tensor, all_seqs, reduction='mean').item()
        # ll_fullseq = F.cross_entropy(sample_probs[0], input_graph.x.argmax(dim=1)[start:end], reduction='mean').item()
        print("score: {}".format(ll_fullseq))
        f.write("score: {}\n".format(ll_fullseq))
        print("mean recovery rate:", np.mean(recovery_list))
        f.write(
            "mean recovery rate: {}\n".format(np.mean(recovery_list))
        )
    return sample_graphs, sample_probs

    # prob,sample_graph = diffusion.ema_model.ddim_sample(input_graph,diverse=True)
    # recovery = (prob.argmax(dim=1) == input_graph.x.argmax(dim = 1)).sum()/(input_graph.x.argmax(dim=1) == input_graph.x.argmax(dim = 1)).shape[0]
    # print('sample sequence: ', ''.join([amino_acids_type[i] for i in sample_graph.argmax(dim=1).tolist()]))
    # print('sample sequence with diveristy mode recovery rate: ',recovery.item())

sample_num = 50
step_interval = 50 #50 #100
multiple_pdbs = False #False
all_sample_graphs = []
sample_graphs_list = []
pdb_max_idx = 10 if multiple_pdbs else 1
# folder_name = "output_1_12_wo_diverse"
folder_name = "output_6m71_wo_diverse_n"
save_folder = "../{}/{}_{}".format(folder_name, pdb_max_idx, sample_num)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
ali_folder = os.path.join(save_folder, "step_interval_{}".format(step_interval))
os.makedirs(ali_folder, exist_ok=True)
if multiple_pdbs:
    ignore_chains = ["A"]
    pdb_folder = "/scratch/network/yy1325/data/sarscov2-15235449-peptide-A-no-water-no-ion-0000-pdbs"
    for i in range(pdb_max_idx):
        pdb_path = os.path.join(pdb_folder, "{:0>4}.pdb".format(i))
        if os.path.exists(pdb_path):
            sample_graphs, sample_probs = sample_sequence(
                pdb_path=pdb_path, ali_folder=ali_folder, ignore_chains=ignore_chains, sample_num=sample_num, step_interval=step_interval, #sta_end=[53, 53+784]
                batch_size = 1,
                )
            all_sample_graphs.extend(sample_graphs)
            sample_graphs_list.append(sample_graphs)
        else:
            print(pdb_path, "does not exist!")
else:
    ignore_chains = ["B", "C", "D"]
    pdb_path = "/scratch/network/yy1325/data/6m71.pdb"
    # pdb_path = '../dataset/raw/test/3fkf.A.pdb'
    # pdb_path = '../dataset_src/all/2lkl.pdb'
    # pdb_path = '../dataset_src/all/1of5.pdb'
    if os.path.exists(pdb_path):
        sample_graphs, sample_probs = sample_sequence(
            pdb_path=pdb_path, ali_folder=ali_folder, ignore_chains=ignore_chains, sample_num=sample_num, step_interval=step_interval, sta_end=[53, 53+784]
            )
        all_sample_graphs.extend(sample_graphs)
        sample_graphs_list.append(sample_graphs)
    else:
        print(pdb_path, "does not exist!")

# num_graphs = len(all_sample_graphs)
# recovery_rate_matrix = np.zeros((num_graphs, num_graphs))
# for i in range(num_graphs):
#     for j in range(num_graphs):
#         recovery = (all_sample_graphs[i].argmax(dim=1) == all_sample_graphs[j].argmax(dim=1)).sum()/(all_sample_graphs[i]).shape[0]
#         recovery_rate_matrix[i, j] = recovery
# np.save(os.path.join(ali_folder, "cmp.npy"), recovery_rate_matrix)
if multiple_pdbs:
    was_dis_matrix = np.zeros((pdb_max_idx, pdb_max_idx))
    mean_rec_matrix = np.zeros((pdb_max_idx, pdb_max_idx))
    for i in range(pdb_max_idx):
        for j in range(i, pdb_max_idx):
            sample_graph_i = sample_graphs_list[i]
            sample_graph_j = sample_graphs_list[j]
            recovery_rate_matrix =  np.zeros((len(sample_graph_i), len(sample_graph_j)))
            assert recovery_rate_matrix.shape == (sample_num, sample_num)
            for k in range(sample_num):
                for l in range(sample_num):
                    recovery = (sample_graph_i[k].argmax(dim=1) == sample_graph_j[l].argmax(dim=1)).sum()/(sample_graph_j[l]).shape[0]
                    recovery_rate_matrix[k, l] = recovery
            D = 1 - recovery_rate_matrix

            prob1 = prob2 = np.ones(sample_num) / sample_num
            dis = ot.emd2(prob1, prob2, D)
            # np.save(os.path.join(ali_folder, "D.npy"), D)
            print(D)
            print(np.mean(D))
            mean_rec_matrix[i, j] = np.mean(recovery_rate_matrix)
            print("Wassertein Distance: {}".format(dis))
            was_dis_matrix[i, j] = dis
    was_dis_matrix =  was_dis_matrix.T + was_dis_matrix - np.diag(np.diag(was_dis_matrix))
    print(was_dis_matrix)
    np.save(save_folder + "/was_dis_mat.npy", was_dis_matrix)

    mean_rec_matrix =  mean_rec_matrix.T + mean_rec_matrix - np.diag(np.diag(mean_rec_matrix))
    print(mean_rec_matrix)
    np.save(save_folder + "/mean_rec_mat.npy", mean_rec_matrix)
    print("trial ends!")

import os
import numpy as np
import torch
from tqdm import tqdm
frame_num = 10
# ProteinMPNN
fa_folder = "ProteinMPNN/outputs/sarscov2-15235449-peptide-A-no-water-no-ion-0001-outputs/seqs_10_100"
assert os.path.exists(fa_folder)
score_list = []
seq_rec_list = []
for i in range(frame_num):
    fa_file = os.path.join(fa_folder, "{:0>4}.fa".format(i))
    with open(fa_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith(">T"):
                continue
            line_sp = line.split(",")
            score = eval(line_sp[2].split("=")[1])
            seq_rec = eval(line_sp[-1].split("=")[1])
            score_list.append(score)
            seq_rec_list.append(seq_rec)
print(len(score_list))
print(len(seq_rec_list))
print("mean seq rec", np.mean(seq_rec_list))
print("mean perplexity", np.exp(np.mean(score_list)))
was_file = "ProteinMPNN/outputs/sarscov2-15235449-peptide-A-no-water-no-ion-0001-outputs/seqs_10_100/was_dis_mat.npy"
was_matrix = np.load(was_file)
tri = np.triu(was_matrix)
assert np.sum(tri>0) == frame_num * (frame_num - 1) / 2
mean_was = np.sum(tri) / np.sum(tri>0)
print("mean was", mean_was)

rec_file = "ProteinMPNN/outputs/sarscov2-15235449-peptide-A-no-water-no-ion-0001-outputs/seqs_10_100/mean_rec_mat.npy"
rec_matrix = np.load(rec_file)

tri = np.triu(rec_matrix) - np.diag(np.diag(rec_matrix))
assert np.sum(tri>0) == frame_num * (frame_num - 1) / 2
mean_seq_rec = np.sum(tri) / np.sum(tri>0)
print("mean seq rec", mean_seq_rec)

# GraDe-IF
fa_folder = "GraDe_IF/output_1/10_50/step_interval_50"
assert os.path.exists(fa_folder)
score_list = []
seq_rec_list = []
for i in range(frame_num):
    fa_file = os.path.join(fa_folder, "{:0>4}.fa".format(i))
    with open(fa_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith(">Sample"):
                continue
            line_sp = line.split(",")
            # score = eval(line_sp[2].split("=")[1])
            seq_rec = eval(line_sp[-1].split("=")[1])
            # score_list.append(score)
            seq_rec_list.append(seq_rec)
# print(len(score_list))
print("---------\nGrade-IF")
print(len(seq_rec_list))
print("mean seq rec", np.mean(seq_rec_list))
# print("mean perplexity", np.exp(np.mean(score_list)))
was_file = "GraDe_IF/output_1/10_50/was_dis_mat.npy"
was_matrix = np.load(was_file)
tri = np.triu(was_matrix)
mean_was = np.sum(tri) / np.sum(tri>0)
print("mean was", mean_was)

rec_file = "GraDe_IF/output_1/10_50/mean_rec_mat.npy"
rec_matrix = np.load(rec_file)
tri = np.triu(rec_matrix) - np.diag(np.diag(rec_matrix))
assert np.sum(tri>0) == frame_num * (frame_num - 1) / 2
mean_rec = np.sum(tri) / np.sum(tri>0)
print("mean seq rec", mean_rec)
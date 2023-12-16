import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

results_folder = "/scratch/network/yy1325/trial/Dynamics-Inverse-Folding/ProteinMPNN/outputs_12_1000_20/seqs_n"
was_list_path = os.path.join(results_folder, "was_dis_list.npy")
mean_rec_list_path = os.path.join(results_folder, "mean_rec_list.npy")
was_dis_list = np.load(was_list_path)
mean_rec_list = np.load(mean_rec_list_path)

all_same_rate_list = np.load(os.path.join(results_folder, "all_same_rate_list.npy"))
all_same_cor_rate_list = np.load(os.path.join(results_folder, "all_same_cor_rate_list.npy"))
cor_part_ratio_list = np.load(os.path.join(results_folder, "cor_part_ratio_list.npy"))

t_list = range(0, 1000, 20)
plt.figure()
plt.plot(t_list[1:], was_dis_list[1:])
plt.xlabel("frame t")
plt.ylabel("Wasserstein Distance")
plt.title("Wasserstein Distance compared to 1st frame")
plt.savefig("was_dis_list.png")

plt.figure()
l1, = plt.plot(t_list[1:], mean_rec_list[1:])
plt.xlabel("frame t")
plt.ylabel("Recovery Rate")
plt.title("Recovery Rate compared to 1st frame")
plt.savefig("mean_rec_list.png")

plt.figure()
l2, = plt.plot(t_list[1:], all_same_rate_list[1:])
l3, = plt.plot(t_list[1:], all_same_cor_rate_list[1:])
l4, = plt.plot(t_list[1:], cor_part_ratio_list[1:])
plt.xlabel("frame t")
plt.ylabel("Ratio")
plt.legend([l2, l3, l4], ["same_AA/total_AA", "same_correct_AA/total_AA", "same_correct_AA/same_AA"])
plt.title("The ratio of same AAs among the first t frames")
plt.savefig("retio_list.png")

frame_num = 10
# ProteinMPNN
score_list = []
seq_rec_list = []
fa_file = "/scratch/network/yy1325/trial/Dynamics-Inverse-Folding/ProteinMPNN/outputs/6m71/seqs/6m71.fa"
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
print("mean seq rec", np.mean(seq_rec_list))
print("mean perplexity", np.exp(np.mean(score_list)))

fa_folder = "ProteinMPNN/outputs/sarscov2-15235449-peptide-A-no-water-no-ion-0000-outputs/seqs_10_n"
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
was_file = "ProteinMPNN/outputs/sarscov2-15235449-peptide-A-no-water-no-ion-0000-outputs/seqs_10_n/was_dis_mat.npy"
was_matrix = np.load(was_file)
tri = np.triu(was_matrix)
assert np.sum(tri>0) == frame_num * (frame_num - 1) / 2
mean_was = np.sum(tri) / np.sum(tri>0)
print("mean was", mean_was)

rec_file = "ProteinMPNN/outputs/sarscov2-15235449-peptide-A-no-water-no-ion-0000-outputs/seqs_10_n/mean_rec_mat.npy"
rec_matrix = np.load(rec_file)

tri = np.triu(rec_matrix) - np.diag(np.diag(rec_matrix))
assert np.sum(tri>0) == frame_num * (frame_num - 1) / 2
mean_seq_rec = np.sum(tri) / np.sum(tri>0)
print("mean seq rec", mean_seq_rec)

# GraDe-IF
os.chdir("/scratch/network/yy1325")
fa_folder = "GraDe_IF/output_1_12_wo_diverse/10_50/step_interval_50"
assert os.path.exists(fa_folder)
score_list = []
perplexity_list = []
seq_rec_list = []
for i in range(frame_num):
    fa_file = os.path.join(fa_folder, "{:0>4}.fa".format(i))
    with open(fa_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("accumulated score"):
                score = eval(line.split(":")[1].strip())
                score_list.append(score)
                continue
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
print("mean perplexity", np.exp(np.mean(score_list)))
was_file = "GraDe_IF/output_1_12_wo_diverse/10_50/was_dis_mat.npy"
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


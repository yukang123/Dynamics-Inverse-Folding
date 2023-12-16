import pymol
import os
from pymol import cmd
import numpy as np
from tqdm import tqdm
# 启动PyMOL
# pymol.finish_launching()

folder = "/Users/mac/Desktop/Inverse Folding/Data/DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion/sarscov2-15235449-peptide-A-no-water-no-ion"
# 加载蛋白质结构和运动轨迹（假设是PDB和DCD文件）
pdb_file = os.path.join(folder, "DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion.pdb")
traj_file = os.path.join(folder, "sarscov2-15235449-peptide-A-no-water-no-ion-0000.dcd")
cmd.load(pdb_file) #, "protein")
# cmd.png("trial")
cmd.load_traj(traj_file)
cmd.remove("chain A")

correct_path = "outputs_trial/seqs_100/correct_idx.npy"
correct_idx = np.load(correct_path)
# 设置视角和显示选项
# cmd.viewport(800, 600)
# cmd.bg_color("white")
# cmd.show("cartoon")
# cmd.color("grey", "protein")

# 选择需要标记的氨基酸范围（这里以10到20号氨基酸为例）
# selected_residues = list(range(31, 35))
# correct_idx = np.load("/Users/mac/Desktop/Inverse Folding/Dynamics-Inverse-Folding/ProteinMPNN/outputs_trial/seqs_100/correct_idx.npy")
# correct_idx = correct_idx[:100]

correct_idx = np.load("/Users/mac/Desktop/Inverse Folding/Dynamics-Inverse-Folding/ProteinMPNN/outputs_trial/seqs_1000_20/correct_idx.npy")
frame_list = range(1, 1000+1, 20)
correct_idx = correct_idx[:10]
frame_list = frame_list[:10]
print(max(frame_list))

# 为每一帧设置氨基酸的颜色
# for frame in range(1, cmd.count_frames("protein") + 1):
# for frame in tqdm(range(1, correct_idx.shape[0]+1)):
for i in tqdm(range(0, correct_idx.shape[0])):
    cmd.frame(frame_list[i])
    cmd.color("green", f"chain B")
    selected_residues = np.nonzero(correct_idx[i])[0]
    # 选择特定范围内的氨基酸并为其设置颜色
    for residue in selected_residues:
        cmd.color("red", f"chain B and resi {residue+1}")

    # 保存当前帧为图像文件（可选）
    # image_file = "frame_{}".format(frame)
    # # cmd.png(image_file, width=800, height=600, dpi=300)
    # cmd.png(image_file)

# 保存视频文件
# cmd.mpeg("your_output_video.mp4", mode=1, state=-1, format="mp4", quality=100, fps=30)

# cmd.mset("1 - {}".format(cmd.count_states()))
cmd.mset("1 - {}".format(max(frame_list)))
cmd.mpng('frame', quiet=0, preserve=2)
# cmd.movie.produce("myy_100")
cmd.movie.produce("myy_1000_20_10")



# 关闭PyMOL
cmd.quit()
import pymol
import os
from pymol import cmd
import numpy as np
# 启动PyMOL
# pymol.finish_launching()

folder = "/Users/mac/Desktop/Inverse Folding/Data/DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion/sarscov2-15235449-peptide-A-no-water-no-ion"
# 加载蛋白质结构和运动轨迹（假设是PDB和DCD文件）
pdb_file = os.path.join(folder, "DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion.pdb")
traj_file = os.path.join(folder, "sarscov2-15235449-peptide-A-no-water-no-ion-0000.dcd")
cmd.load(pdb_file) #, "protein")
# cmd.png("trial")
cmd.load_traj(traj_file)

correct_path = "outputs_trial/seqs_100/correct_idx.npy"
correct_idx = np.load(correct_path)
# 设置视角和显示选项
# cmd.viewport(800, 600)
# cmd.bg_color("white")
# cmd.show("cartoon")
# cmd.color("grey", "protein")

# 选择需要标记的氨基酸范围（这里以10到20号氨基酸为例）
selected_residues = list(range(31, 35))

# 为每一帧设置氨基酸的颜色
# for frame in range(1, cmd.count_frames("protein") + 1):
for frame in range(1, 5):
    cmd.frame(frame)
    
    # 选择特定范围内的氨基酸并为其设置颜色
    for residue in selected_residues:
        cmd.color("red", f"chain B and resi {residue}")

    # 保存当前帧为图像文件（可选）
    image_file = "frame_{}".format(frame)
    # cmd.png(image_file, width=800, height=600, dpi=300)
    cmd.png(image_file)

# 保存视频文件
# cmd.mpeg("your_output_video.mp4", mode=1, state=-1, format="mp4", quality=100, fps=30)

# cmd.mset("1 - {}".format(cmd.count_states()))
cmd.mset("1 - {}".format(4))
cmd.mpng('frame', quiet=0, preserve=2)
cmd.movie.produce("trial")


# 关闭PyMOL
cmd.quit()
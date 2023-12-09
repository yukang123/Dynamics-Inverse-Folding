import mdtraj as md
import os
import numpy as np
from tqdm import tqdm
data_folder = "/Users/mac/Desktop/Inverse Folding/Data/DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion/sarscov2-15235449-peptide-A-no-water-no-ion"
traj_name = "sarscov2-15235449-peptide-A-no-water-no-ion-0001.dcd"
traj_file = os.path.join(data_folder, traj_name)
pdb_file = os.path.join(data_folder, "DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion.pdb")
traj = md.load(traj_file, top=pdb_file)
print(traj)

pdb = md.load(pdb_file)
print(pdb)

traj_pdbs_folder = os.path.join(data_folder, os.path.splitext(traj_name)[0]+"-pdbs")
if not os.path.exists(traj_pdbs_folder):
    os.makedirs(traj_pdbs_folder)

dis_list = []
for i in tqdm(range(len(traj))):
    dis_i = np.sum(np.abs(traj[i].xyz[0,0,:] - pdb.xyz[0,0,:]))
    dis_list.append(dis_i)
    if dis_i < 10**-5:
        print(i)
    traj[i].save(os.path.join(traj_pdbs_folder, "{:0>4d}.pdb".format(i)))
print("trial ends!")

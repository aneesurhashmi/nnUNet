#!/bin/bash -eux
#SBATCH --job-name=seg_01
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anees.hashmi@hpi.de
##SBATCH --container-image=nvidia/cuda:11.4.2-base-ubuntu20.04
#SBATCH --partition=gpua100 # -p
##SBATCH --cpus-per-task=3 # -c
##SBATCH --mem=8gb
#SBATCH --gpus=1
#SBATCH --time=12:00:00 
#SBATCH --output=/dhc/home/anees.hashmi/bowel-segmentation/nnUNet/bash_logs/seg_01%j.log # %j is job id

date
pwd
hostname -f
nproc
nvidia-smi

# already set in .bashrc
export nnUNet_raw="/dhc/home/anees.hashmi/data/nnUNet_raw"
export nnUNet_preprocessed="/dhc/home/anees.hashmi/data/nnUNet_preprocessed"
export nnUNet_results="/dhc/home/anees.hashmi/bowel-segmentation/nnUNet/results"


# # # nnUNetv2_train TASK_ID FOLD_ID CONFIGURATION TRAINER
# nnUNetv2_train 001 3d_fullres 3 -tr nnUNetTrainer_250epochs


# SwinUNETR
nnUNetv2_train 001 3d_fullres 3 -tr nnUNetTrainerSwinUNETR    
nnUNetv2_train 001 3d_fullres 4 -tr nnUNetTrainerSwinUNETR    


# for fold in {1..4}
# do
#     # nnUNetv2_train 001 3d_fullres $fold # 3D with full resolution and Fold $fold
#     nnUNetv2_train 001 3d_fullres $fold -tr nnUNetTrainerSwinUNETR    
# done
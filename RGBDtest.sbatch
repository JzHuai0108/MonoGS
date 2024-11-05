#!/bin/bash

#SBATCH --account=youraccount
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=monogs-slurm.out

nvcc --version
nvidia-smi

echo "SLURM_SUBMIT_DIR: " $SLURM_SUBMIT_DIR # this is current working dir

conda init
source ~/.bashrc
cd $SLURM_SUBMIT_DIR
conda activate monogs


python ~/project/code/MonoGS/slam.py --config configs/mono/tum/fr3_office.yaml                                                                              

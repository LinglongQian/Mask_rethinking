#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/ijcai24/overlay_premask/air.log
#SBATCH --gres=gpu:1
#SBATCH --partition=biomed_a100_gpu
#SBATCH --job-name=bash
#SBATCH --mem=256G
#SBATCH --mail-user=linglong.qian@kcl.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=2-00:00:00

module load anaconda3/2021.05-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0
module load cudnn/8.2.4.15-11.4-gcc-13.2.0

nvidia-smi

cat 1>&2 <<END
task ${SLURM_JOB_ID}
END

cd /scratch/users/k1814348/ijcai24/overlay_premask
source activate pypots

python train_models_for_air.py
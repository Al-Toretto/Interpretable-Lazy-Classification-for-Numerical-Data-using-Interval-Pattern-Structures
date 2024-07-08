#!/bin/bash
#
#SBATCH --job-name=ml-classic
#SBATCH --time=3-0:0
#SBATCH --cpus-per-task=10
#SBATCH --output=run.log
#SBATCH --error=error.err
#SBATCH --ntasks=1
module load Python
source activate my_env
srun python classic-ml.py $1


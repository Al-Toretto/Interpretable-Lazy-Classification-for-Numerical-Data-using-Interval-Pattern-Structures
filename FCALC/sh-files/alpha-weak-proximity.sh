#!/bin/bash
#
#SBATCH --job-name=alpha-prox
#SBATCH --time=3-0:0
#SBATCH --output=run.log
#SBATCH --error=error.err
#SBATCH --ntasks=1
module load Python
source activate my_env
srun python alpha-weak-proximity.py $1


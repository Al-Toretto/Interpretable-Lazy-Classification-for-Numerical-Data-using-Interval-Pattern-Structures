#!/bin/bash
#
#SBATCH --job-name=stability
#SBATCH --time=10-0:0
#SBATCH --output=run.log
#SBATCH --error=error.err
#SBATCH --ntasks=1
module load Python
source activate my_env
srun python stability.py $1 $2


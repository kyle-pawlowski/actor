#!/bin/bash
#SBATCH -JRNN_Model_Selection            # Job name
#SBATCH -N1 --ntasks-per-node=4          # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G                 # Memory per core
#SBATCH -t120                             # Duration of the job (Ex: 15 mins)
#SBATCH -ornn_logs/Report-%j.out         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=kpawlowski3@gatech.edu # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                     # Change to working directory

module load anaconda3                    # Load module dependencies
conda activate actor
srun python python/model_selection.py rnn                  

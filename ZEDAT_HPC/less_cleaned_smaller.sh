#!/bin/bash

#SBATCH --partition=gpu 	#use GPU partition
#SBATCH --nodes=1		#number of nodes (max 12 GPU nodes avail)
#SBATCH --gres=gpu:1		#number	of GPUs	per node (max=2)		
#SBATCH --mail-user=bernadeta@zedat.fu-berlin.de
#SBATCH --mail-type=all		
#SBATCH --job-name=exp_less_len140_smaller_nonorm               
#SBATCH --ntasks=1                         
#SBATCH --mem=9G                      
#SBATCH --time=15:00:00                         
#SBATCH --qos=standard  


echo "Running on $(hostname)"

path_to_config_file="/home/bernadeta/BA_code/scripts/slurm/config_files/exp_less_len140_smaller.yaml"
path_to_model_dir="/home/bernadeta/BA_code/models/exp_less_len140_smaller_nonorm"

JOB_ID=$SLURM_JOB_ID

module add Python/3.8.6-GCCcore-10.2.0

# Activate the environment
source /home/bernadeta/python-venvs/joeynmt/bin/activate

# Train the model
python3 -m joeynmt train ${path_to_config_file} 

mv "slurm-${JOB_ID}.out" "${path_to_model_dir}"
#!/bin/bash

#SBATCH --partition=gpu 	#use GPU partition
#SBATCH --nodes=1		#number of nodes (max 12 GPU nodes avail)
#SBATCH --gres=gpu:1		#number	of GPUs	per node 		
#SBATCH --mail-user=myemail
#SBATCH --mail-type=all		
#SBATCH --job-name=myjobname
#SBATCH --ntasks=1                         
#SBATCH --mem=3G                      
#SBATCH --time=00:20:00     #testing takes around 10 min on 2000 sentence pairs, one GPU                
#SBATCH --qos=standard  


echo "Running on $(hostname)"

path_to_config_file="path/to/config/file"
path_to_model_dir="path/to/model/dir/"
path_to_output_file="path/to/output/file"

JOB_ID=$SLURM_JOB_ID

module add Python/3.8.6-GCCcore-10.2.0

# Activate the environment
source /path/to/new/virtual/environment/bin/activate

# Train the model
python3 -m joeynmt test ${path_to_config_file} --output_path ${path_to_output_file}

mv "slurm-${JOB_ID}.out" "${path_to_model_dir}"
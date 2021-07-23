#!/bin/bash

#SBATCH --partition=gpu 	#use GPU partition
#SBATCH --nodes=1		#number of nodes (max 12 GPU nodes avail)
#SBATCH --gres=gpu:1		#number	of GPUs	per node (max=2)		
#SBATCH --mail-user=bernadeta@zedat.fu-berlin.de
#SBATCH --mail-type=all		
#SBATCH --job-name=bicleaner_cleaned_300_test              
#SBATCH --ntasks=1                         
#SBATCH --mem=4G                      
#SBATCH --time=01:00:00                         
#SBATCH --qos=standard  


echo "Running on $(hostname)"

path_to_config_file="/home/bernadeta/BA_code/scripts/slurm/config_files/bicleaner_cleaned_8.7.21.yaml"
path_to_model_dir="/home/bernadeta/BA_code/models/bicleaner_cleaned_13.7.21_len300"

JOB_ID=$SLURM_JOB_ID

module add Python/3.8.6-GCCcore-10.2.0

# Activate the environment
source /home/bernadeta/python-venvs/joeynmt/bin/activate

# Train or test the model
python3 -m joeynmt translate ${path_to_config_file} < /home/bernadeta/BA_code/data/02-preprocessed/newstest2016/wmt16test.8000.bpe.en > out.biclean300.wmt16.test

#mv "slurm-${JOB_ID}.out" "${path_to_model_dir}"
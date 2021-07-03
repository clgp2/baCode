#!/bin/bash

#SBATCH --partition=gpu 	#use GPU partition
#SBATCH --nodes=1		#number of nodes (max 12 GPU nodes avail)
#SBATCH --gres=gpu:1		#number	of GPUs	per node (max=2?)		
#SBATCH --job-name=test-fulldata-transformer                   
#SBATCH --mail-user=bernadeta@zedat.fu-berlin.de
#SBATCH --mail-type=all		
#SBATCH --ntasks=1                         
#SBATCH --mem=5G                      
#SBATCH --time=00:50:00                         
#SBATCH --qos=standard                          

path_to_config_file="/home/bernadeta/BA_code/scripts/slurm/config_files/transformer_enro_dcep_02-01.yaml"
path_to_model_dir="/home/bernadeta/BA_code/models/02-01-preprocessed/beamsize5/"

JOB_ID=$SLURM_JOB_ID

srun echo $CUDA_VISIBLE_DEVICES
python --version

module add Python/3.8.6-GCCcore-10.2.0
python --version

source /home/bernadeta/python-venvs/joeynmt/bin/activate
python --version
	
#env

python3 -m joeynmt test ${path_to_config_file} --output_path=${path_to_model_dir}

mv "slurm-${JOB_ID}.out" "${path_to_model_dir}/"
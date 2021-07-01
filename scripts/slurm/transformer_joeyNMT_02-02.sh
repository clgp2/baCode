#!/bin/bash

#SBATCH --partition=gpu 	#use GPU partition
#SBATCH --nodes=1		#number of nodes (max 12 GPU nodes avail)
#SBATCH --gres=gpu:1		#number	of GPUs	per node (max=2?)		
#SBATCH --job-name=bicleaner-transformer                   
#SBATCH --mail-user=bernadeta@zedat.fu-berlin.de
#SBATCH --mail-type=all		
#SBATCH --ntasks=1                         
#SBATCH --mem=20G                      
#SBATCH --time=30:00:00                         
#SBATCH --qos=standard                          

path_to_config_file=/home/bernadeta/bachelorarbeit/src/joeynmt/transformer_base
path_to_model_dir=/home/bernadeta/bachelorarbeit/experiments/bicleaner-tok-bpe/beamsize5

srun echo $CUDA_VISIBLE_DEVICES
python --version

module add Python/3.8.6-GCCcore-10.2.0
python --version

source /home/bernadeta/python-venvs/joeynmt/bin/activate
python --version
	
#env

python3 -m joeynmt train transformer_enro_dcep.yaml


#!/bin/bash

#SBATCH --partition=gpu 	#use GPU partition
#SBATCH --nodes=1		#number of nodes (max 12 GPU nodes avail)
#SBATCH --gres=gpu:2		#number	of GPUs	per node (max=2?)		
#SBATCH --job-name=bahdanau_justBPE_140                   
#SBATCH --mail-user=bernadeta@zedat.fu-berlin.de
#SBATCH --mail-type=all		
#SBATCH --ntasks=1                         
#SBATCH --mem=30G                      
#SBATCH --time=30:00:00                         
#SBATCH --qos=standard                          

srun echo $CUDA_VISIBLE_DEVICES
python --version

module add Python/3.8.6-GCCcore-10.2.0
python --version

source /home/bernadeta/python-venvs/joeynmt/bin/activate
python --version
	
#env

python3 -m joeynmt train /home/bernadeta/venv_docs/joeynmt/bahdanau/bahdanau_enro_dcep.yaml --output_path /home/bernadeta/models/new/bahdanau_justBPE_140


#!/bin/bash

#SBATCH --partition=gpu 	#use GPU partition
#SBATCH --nodes=1		#number of nodes (max 12 GPU nodes avail)
#SBATCH --gres=gpu:1		#number	of GPUs	per node (max=2?)		
#SBATCH --mail-user=bernadeta@zedat.fu-berlin.de
#SBATCH --mail-type=all		
#SBATCH --job-name=opennmt-test                 
#SBATCH --ntasks=1                         
#SBATCH --mem=5G                      
#SBATCH --time=05:00:00                         
#SBATCH --qos=standard  


echo "Running on $(hostname)"

path_to_model_dir="/home/bernadeta/BA_code/models/opennmt"

JOB_ID=$SLURM_JOB_ID

module add Python/3.8.6-GCCcore-10.2.0

# Activate the environment
source /home/bernadeta/python-venvs/opennmt-py/bin/activate

# Train the model
onmt_translate -model /home/bernadeta/BA_code/scripts/slurm/exp/model.en-ro_step_40000.pt -src /home/bernadeta/BA_code/data/02-preprocessed/02-02-bicleaner-preprocessed/test.30000.bpe.en -output /home/bernadeta/BA_code/models/opennmt/pred_40000.txt -gpu 0 -verbose

mv "slurm-${JOB_ID}.out" "${path_to_model_dir}/"
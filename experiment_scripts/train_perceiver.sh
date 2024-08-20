#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:264754907
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=26:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=train_perceiver_conceptualCaptions
#SBATCH --output=terminal.txt

source VLPs/bin/activate
export WANDB_API_KEY=4af4c9a807cc6c8b208f4388909fc7f9ead825f0
python3 train_perceiver.py --dataset ConceptualCaptions --batch_size 128 --no_epochs 50
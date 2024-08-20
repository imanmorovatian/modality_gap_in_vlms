#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:264754907
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=compute_embeddings
#SBATCH --output=terminal.txt

source VLPs/bin/activate
export WANDB_API_KEY=4af4c9a807cc6c8b208f4388909fc7f9ead825f0
python3 compute_embeddings.py --model Perceiver --dataset conceptualCaptions --batch_size 128 --captions_per_image 5
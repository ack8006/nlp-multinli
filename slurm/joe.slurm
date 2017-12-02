#!/bin/bash
#
#SBATCH --job-name=multisnli
#SBATCH --tasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem=15000
#SBATCH --gres=gpu:1
#SBATCH --output=exp_%A.out
#SBATCH --error=exp_%A.err

# Log what we're running and where
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

cd /scratch/jcv312/nlp-multinli/models

module purge
module load python3/intel/3.5.3
module load nltk/python3.5/3.2.4

python3 -m pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl --user
python3 -m pip install https://github.com/pytorch/text/archive/master.zip --user

python3 -u main.py --cuda --model_type DA --max_vocab_size 15000 --n_epochs 100 --batch_size 64 --d_embed 300 --d_hidden 350 --lr 0.0001 --dropout_mlp 0.2 --word_vectors glove.6B.300d --dev_every 550 --save_model --intra_sentence TRUE

#python main.py --model_type DA --max_vocab_size 1000 --n_epochs 15 --batch_size 4 --d_embed 50 --d_hidden 10 --lr 0.05 --dropout_mlp 0.2 --word_vectors glove.6B.50d --dev_every 10

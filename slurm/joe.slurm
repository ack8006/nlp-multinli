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

cd /scratch/act444/github/nlp-snli/models	

module purge
module load python3/intel/3.5.3
module load nltk/python3.5/3.2.4

python3 -m pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl --user
python3 -m pip install https://github.com/pytorch/text/archive/master.zip --user

python3 -u main.py --cuda --model_type ConcatModel --max_vocab_size 15000 --n_epochs 15 --batch_size 64 --d_embed 300 --d_hidden 200 --n_layers 1 --lr 0.001 --dropout_rnn 0.2 --dropout_mlp 0.5 --word_vectors glove.840B.300d --bidir --dev_every 1550

#python main.py --model_type ESIM --max_vocab_size 100 --n_epochs 15 --batch_size 10 --d_embed 50 --d_hidden 10 --n_layers 1 --lr 0.001 --dropout_rnn 0.2 --dropout_mlp 0.5 --word_vectors glove.6B.50d --bidir --dev_every 10
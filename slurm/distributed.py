import subprocess

priors = [
    [64, 200, 5e-4, 0.2, False],
    [64, 200, 1e-4, 0.2, False],
    [128, 200, 1e-4, 0.2, False],
    [64, 200, 5e-5, 0.2, False],
    [128, 200, 5e-5, 0.2, False],
    [64, 200, 5e-4, 0.3, False],
    [64, 300, 5e-4, 0.5, False],
    [64, 200, 1e-4, 0.2, True]
]

for batch_size in [64]:
    for d_hidden in [150, 200, 250]:
        for lr in [5e-5, 1e-4, 5e-4]:
            for dropout_mlp in [0.1, 0.2, 0.3]:
                for intra_sentence in [True, False]:

                    if [batch_size, d_hidden, lr, dropout_mlp, intra_sentence] in priors:
                        continue

                    slurm_text = """#!/bin/bash
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

python3 -u main.py --cuda --model_type DA --max_vocab_size 15000 --n_epochs 100 --batch_size {} --d_embed 300 --d_hidden {} --lr {} --dropout_mlp {} --word_vectors glove.6B.300d --dev_every 550 --save_model --intra_sentence {}

""".format(batch_size, d_hidden, lr, dropout_mlp, intra_sentence)

                text_file = open("distrib_temp.slurm", "wb")
                text_file.write("%s" % slurm_text)
                text_file.close()

                subprocess.call("sbatch ./distrib_temp.slurm", shell=True)
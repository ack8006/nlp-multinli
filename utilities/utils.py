import os
from argparse import ArgumentParser
from torchtext import data


FILES = {
    'train': ('multinli_1.0_train.txt', 'train.tsv'),
    # 'train': ('multinli_1.0_dev_matched.txt', 'val_match.tsv'),
    'val_matched': ('multinli_1.0_dev_matched.txt', 'val_match.tsv'),

    # 'train': ('snli_1.0_train.txt', 'snli_train.tsv'),
    # 'val_matched': ('snli_1.0_dev.txt', 'snli_dev.tsv'),

    'val_mismatched': ('multinli_1.0_dev_mismatched.txt', 'val_mismatch.tsv')
}


def tsv_file_existence_check(file_txt, file_tsv):
    if not os.path.exists('../data'):
        os.mkdir('../data/')
        raise Exception('Please download multinli files into the nlp-snli/data/ directory')
    else:
        assert os.path.exists('../data/{}'.format(file_txt)) or os.path.exists('../data/{}'.format(file_tsv))
    if file_tsv not in os.listdir('../data'):
        print('Generating {} file'.format(file_tsv))
        with open('../data/{}'.format(file_txt), 'r') as f_txt:
            lines = f_txt.readlines()
            with open('../data/{}'.format(file_tsv), 'w') as f_tsv:
                for line in lines[1:]:
                    line = line.split('\t')
                    if line[0] == '-':  # Examples without a Gold Consensus
                        continue
                    f_tsv.write('\t'.join([line[0], line[5], line[6]]) + '\n')


def get_dataset(text_field, label_field, dataset):
    if dataset not in FILES.keys():
        raise AttributeError('Please set dataset as either {}'.format(FILES.keys()))
    file_txt, file_tsv = FILES[dataset]
    tsv_file_existence_check(file_txt, file_tsv)
    print('Creating {} Dataset'.format(dataset.title()))

    dataset = data.TabularDataset(path='../data/{}'.format(file_tsv),
                                  format='TSV',
                                  fields=[('label', label_field),
                                          ('premise', text_field),
                                          ('hypothesis', text_field)])
    return dataset


def get_args():
    parser = ArgumentParser(description='PyTorch MultiNLI Model')
    parser.add_argument('--dataset', type=str, default='multinli')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--val_set', type=str, default='val_matched',
                        help='Which Val Set (val_matched, val_mismatched')
    parser.add_argument('--max_vocab_size', type=int, default=20000)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--intra_sentence', type=bool, default=False)
    parser.add_argument('--sentence_len', type=int, default=None)
    parser.add_argument('--d_embed', type=int, default=200)
    # parser.add_argument('--d_proj', type=int, default=200)
    parser.add_argument('--d_hidden', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--n_linear_layers', type=int, default=3)
    parser.add_argument('--mp_dim', type=int, default=15)
    parser.add_argument('--agg_d_hidden', type=int, default=100)
    parser.add_argument('--agg_n_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout_rnn', type=float, default=0.0)
    parser.add_argument('--dropout_mlp', type=float, default=0.0)
    parser.add_argument('--dropout_emb', type=float, default=0.0,
                        help='Dropout applied to the embeddings')
    parser.add_argument('--word_vectors', type=str, default='glove.6B.200d')
    parser.add_argument('--bidir', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--no_comet', action='store_true')
    parser.add_argument('--dev_every', type=int, default=300)
    parser.add_argument('--load_model', type=str, default='')
    args = parser.parse_args()
    return args

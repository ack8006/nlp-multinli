import os
from torchtext import data


FILES = {'train': ('multinli_1.0_train.txt', 'train.tsv'),
         'val_matched': ('multinli_1.0_dev_matched.txt', 'val_match.tsv'),
         'val_mismatched': ('multinli_1.0_dev_mismatched.txt', 'val_mismatch.tsv')}


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

from argparse import ArgumentParser
import sys

import torch
import torch.nn as nn
from torchtext import data

from models import ConcatModel
sys.path.append('../utilities')
import utils, tokenizers #get_dataset
# from tokenizers import custom_tokenizer


MODELS = {'ConcatModel': ConcatModel}


def main():
    parser = ArgumentParser(description='PyTorch MultiNLI Model')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--val_set', type=str, default='val_matched',
                        help='Which Val Set (val_matched, val_unmatched')
    parser.add_argument('--max_vocab_size', type=int, default=20000)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sentence_len', type=int, default=25)
    parser.add_argument('--d_embed', type=int, default=200)
    parser.add_argument('--d_proj', type=int, default=200)
    parser.add_argument('--d_hidden', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--word_vectors', type=str, default='glove.6B.200d')
    parser.add_argument('--bidir', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--load_model', type=str, default='')
    args = parser.parse_args()

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    text_field = data.Field(tokenize=tokenizers.custom_tokenizer,
                            fix_length=args.sentence_len,
                            unk_token='<**UNK**>')
    label_field = data.Field(sequential=False, unk_token=None)

    train = utils.get_dataset(text_field, label_field, 'train')
    val = utils.get_dataset(text_field, label_field, args.val_set)

    text_field.build_vocab(train, max_size=args.max_vocab_size)
    label_field.build_vocab(train, val)

    if args.word_vectors:
        text_field.vocab.load_vectors(args.word_vectors)

    device = -1
    if args.cuda:
        device = 1

    print('Generating Iterators')
    train_iter, val_iter = data.BucketIterator.splits((train, val),
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                sort_key=sort_key,
                                                device=device)

    args.n_embed = len(text_field.vocab)
    args.d_out = len(label_field.vocab)

    if args.load_model:
        model = torch.load(args.load_model)
    else:
        model = MODELS[args.model_type](args)
        print('Loading Word Embeddings')
        model.embed.weight.data = text_field.vocab.vectors

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda():
        model.cuda()

    print('Training Model')

    for epoch in range(1, args.n_epochs + 1):
        for batch_ind, batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.label)

            # calculate accuracy of predictions in the current batch
            n_correct += (torch.max(out, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total
            print(train_acc)

            loss.backward()
            optimizer.step()

            model.eval()






if __name__ == '__main__':
    main()

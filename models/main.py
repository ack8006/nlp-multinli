import sys

from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import nltk

from models import ConcatModel, CosineModel, ESIM
sys.path.append('../utilities')
from tokenizers import custom_tokenizer
from utils import get_dataset, get_args

nltk.download('punkt')

MODELS = {'ConcatModel': ConcatModel,
          'CosineModel': CosineModel,
          'ESIM': ESIM}


def sort_key(ex):
    return data.interleave_keys(len(ex.premise), len(ex.hypothesis))


def main():

    args = get_args()
    experiment = Experiment(api_key="5yzCYxgDmFnt1fhJWTRQIkETT", log_code=True)
    hyperparams = vars(args)
    experiment.log_multiple_params(hyperparams)

    text_field = data.Field(tokenize=custom_tokenizer,
                            fix_length=args.sentence_len,
                            unk_token='<**UNK**>')
    label_field = data.Field(sequential=False, unk_token=None)

    train = get_dataset(text_field, label_field, 'train')
    val = get_dataset(text_field, label_field, args.val_set)

    text_field.build_vocab(train, max_size=args.max_vocab_size)
    label_field.build_vocab(train, val)

    if args.word_vectors:
        text_field.vocab.load_vectors(args.word_vectors)

    device = -1
    if args.cuda:
        device = None

    print('Generating Iterators')
    train_iter, val_iter = data.BucketIterator.splits((train, val),
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                sort_key=sort_key,
                                                device=device)
    train_iter.repeat = False

    args.n_embed = len(text_field.vocab)
    args.d_out = len(label_field.vocab)
    args.n_cells = args.n_layers
    if args.bidir:
        args.n_cells *= 2
    print(args)

    if args.load_model:
        model = torch.load(args.load_model)
    else:
        model = MODELS[args.model_type](args)
        print('Loading Word Embeddings')
        model.embed.weight.data = text_field.vocab.vectors

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print(model)

    print('Training Model')

    best_val_acc = 0.0
    for epoch in range(1, args.n_epochs + 1):
        train_iter.init_epoch()
        for batch_ind, batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.label)
            loss.backward()
            optimizer.step()

            if (batch_ind != 0) and (batch_ind % args.dev_every == 0):
                val_correct, val_loss = evaluate(val_iter, model, criterion)
                val_accuracy = 100 * val_correct / len(val)
                print('    Batch Step {}/{}, Val Loss: {:.4f}, Val Accuracy: {:.4f}'.\
                            format(batch_ind,
                                   len(train) // args.batch_size,
                                   val_loss,
                                   val_accuracy))

        train_correct, train_loss = evaluate(train_iter, model, criterion)
        val_correct, val_loss = evaluate(val_iter, model, criterion)
        val_accuracy = 100 * val_correct / len(val)

        experiment.log_metric("Train loss", train_loss)
        experiment.log_metric("Val loss", val_loss)
        experiment.log_metric("Accuracy (val)", val_accuracy)

        if args.save_model and (val_accuracy > best_val_acc):
            best_val_acc = val_accuracy
            if best_val_acc > 60:
                snapshot_path = '../saved_models/Model_{}_acc_{:.4f}_epoch_{}_model.pt'.format(args.model_type, val_accuracy, epoch)
                torch.save(model.cpu(), snapshot_path)

        print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.2f}, Val Acc: {:.2f}, Best Val Acc: {:.2f}'.\
                format(epoch,
                       train_loss,
                       val_loss,
                       100 * train_correct / len(train),
                       val_accuracy,
                       best_val_acc))


def evaluate(iterator, model, criterion):
    model.eval()
    n_correct, eval_losses = 0, []
    for batch_ind, batch in enumerate(iterator):
        out = model(batch)
        n_correct += (torch.max(out, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        eval_losses.append(criterion(out, batch.label).data[0])
    eval_loss = sum(eval_losses) / len(eval_losses)
    return n_correct, eval_loss


if __name__ == '__main__':
    main()

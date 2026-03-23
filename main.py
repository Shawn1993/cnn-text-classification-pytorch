#! /usr/bin/env python
import os
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
import model
import train
import mydatasets


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-dataset', type=str, default='MR', help='dataset to use: MR or SST [default: MR]')
parser.add_argument('-no-phrases', action='store_true', default=False, help='SST: use sentence-level only (no phrase data)')
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-optimizer', type=str, default='adam', help='optimizer: adam or adadelta [default: adam]')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# load dataset
print("\nLoading data...")
if args.dataset == 'SST':
    use_phrases = not args.no_phrases
    train_dataset, dev_dataset, test_dataset, text_vocab, label_vocab = mydatasets.SSTDataset.splits(
        use_phrases=use_phrases)
    print(f"SST-1 (5-class, phrases={use_phrases}): train={len(train_dataset)}, dev={len(dev_dataset)}, test={len(test_dataset)}")
else:
    train_dataset, dev_dataset, text_vocab, label_vocab = mydatasets.MRDataset.splits()
    test_dataset = None

train_iter = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=mydatasets.collate_fn)
dev_iter = DataLoader(dev_dataset, batch_size=len(dev_dataset),
                      shuffle=False, collate_fn=mydatasets.collate_fn)
if test_dataset is not None:
    test_iter = DataLoader(test_dataset, batch_size=len(test_dataset),
                           shuffle=False, collate_fn=mydatasets.collate_fn)
else:
    test_iter = None


# update args and print
args.embed_num = len(text_vocab)
args.class_num = len(label_vocab)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot, weights_only=True))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()


# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_vocab, label_vocab, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        print("\n--- Dev Set ---")
        train.eval(dev_iter, cnn, args)
        if test_iter is not None:
            print("--- Test Set ---")
            train.eval(test_iter, cnn, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't exist.\n")
else:
    print()
    try:
        best_acc = train.train(train_iter, dev_iter, cnn, args)
        if test_iter is not None:
            # Load best model for test evaluation
            import glob
            best_files = glob.glob(os.path.join(args.save_dir, 'best_steps_*.pt'))
            if best_files:
                best_path = max(best_files, key=os.path.getmtime)
                print(f'\nLoading best model from {best_path}...')
                cnn.load_state_dict(torch.load(best_path, weights_only=True))
            print("\n--- Final Test Set Evaluation ---")
            train.eval(test_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

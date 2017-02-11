import argparse
import torch
import torchtext.data as data
import torchtext.datasets as datasets

import model
import train
import mydatasets


parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-batch-size', type=int, default=64, metavar='N', help='batch size for training [default: 50]')
parser.add_argument('-lr', type=float, default=0.001, metavar='LR', help='initial learning rate [default: 0.01]')
parser.add_argument('-epochs', type=int, default=200, metavar='N', help='number of epochs for train [default: 10]')
parser.add_argument('-dropout', type=float, default=0.5, metavar='', help='the probability for dropout [default: 0.5]')
parser.add_argument('-max_norm', type=float, default=3.0, help='l2 constraint of parameters')
parser.add_argument('-cpu', action='store_true', default=False, help='disable the gpu' )
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data')
# model
parser.add_argument('-embed-dim', type=int, default=128)
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='Comma-separated kernel size to use for convolution')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-class-num', type=int, default=2, help='number of class')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
parser.add_argument('-num-workers', type=int, default=0, help='how many subprocesses to use for data loading [default: 0]')
# log
parser.add_argument('-log-interval', type=int, default=1, help='how many batches to wait before logging training status')
parser.add_argument('-test-interval', type=int, default=100, help='how many epochs to wait before testing')
parser.add_argument('-save-interval', type=int, default=500, help='how many epochs to wait before saving')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-save-dir', type=str, default='', help='where to save the checkpoint')
args = parser.parse_args()
args.cuda = not args.cpu and torch.cuda.is_available()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

# load SST
'''
print("Loading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
text_field.build_vocab(train_data, dev_data, test_data)
label_field.build_vocab(train_data)
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                    (train_data, dev_data, test_data), 
                                    batch_sizes=(args.batch_size, 
                                                 len(dev_data), 
                                                 len(test_data)),
                                    device=-1, repeat=False)
'''

# load MR
print("Loading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
text_field.build_vocab(train_data, dev_data)
label_field.build_vocab(train_data)
train_iter, dev_iter = data.Iterator.splits((train_data, dev_data), 
                                    batch_sizes=(args.batch_size, len(dev_data)),
                                    device=-1, repeat=False)
# args
args.embed_num = len(text_field.vocab)
# model
cnn = model.CNN_Text(args)
# train
if args.test: 
    if args.snapshot is None:
        train.test(test_iter, cnn, args)
else: 
    train.train(train_iter, dev_iter, cnn, args)


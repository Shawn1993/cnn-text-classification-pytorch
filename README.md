## Introduction
This is the implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.

1. Kim's implementation of the model in Theano:
[https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)
2. Denny Britz has an implementation in Tensorflow:
[https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
3. Alexander Rakhlin's implementation in Keras;
[https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)

## Requirement
* python >= 3.8
* pytorch >= 2.0

No longer depends on `torchtext` or `numpy`. Data loading uses standard PyTorch `Dataset` and `DataLoader`.

## Result

Two datasets were tested: MR and SST.

|Dataset|Class Size|Best Result|Kim's Paper Result|
|---|---|---|---|
|MR|2|76.5%(CNN-rand)|76.1%(CNN-rand)|
|SST|5|45.6%(CNN-rand)|45.0%(CNN-rand)|

Both results are consistent with Kim's paper. The SST result uses phrase-level training data, Adadelta optimizer, and embedding dimension 300, matching the original paper's setup:

```
python3 main.py -dataset SST -embed-dim 300 -batch-size 50 -optimizer adadelta -lr 1.0 -epochs 15 -early-stop 10000 -test-interval 500
```

## Usage
```
python3 main.py -h
```

You will get:

```
CNN text classificer

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                initial learning rate [default: 0.001]
  -epochs N             number of epochs for train [default: 256]
  -batch-size N         batch size for training [default: 64]
  -log-interval N       how many steps to wait before logging training status [default: 1]
  -test-interval N      how many steps to wait before testing [default: 100]
  -save-interval N      how many steps to wait before saving [default: 500]
  -save-dir DIR         where to save the snapshot
  -early-stop N         iteration numbers to stop without performance increasing [default: 1000]
  -save-best BOOL       whether to save when get best performance [default: True]
  -dataset DATASET      dataset to use: MR or SST [default: MR]
  -no-phrases           SST: use sentence-level only (no phrase data)
  -shuffle              shuffle the data every epoch
  -dropout DROPOUT      the probability for dropout [default: 0.5]
  -max-norm FLOAT       l2 constraint of parameters [default: 3.0]
  -embed-dim N          number of embedding dimension [default: 128]
  -kernel-num N         number of each kind of kernel [default: 100]
  -kernel-sizes STR     comma-separated kernel size to use for convolution [default: 3,4,5]
  -static               fix the embedding
  -optimizer OPTIMIZER  optimizer: adam or adadelta [default: adam]
  -device DEVICE        device to use for iterate data, -1 mean cpu [default: -1]
  -no-cuda              disable the gpu
  -snapshot FILE        filename of model snapshot [default: None]
  -predict TEXT         predict the sentence given
  -test                 train or test
```

## Train
```
python3 main.py
```
You will get:

```
Batch[100] - loss: 0.655424  acc: 59.3750%
Evaluation - loss: 0.672396  acc: 57.6923%(615/1066)
```

## Test
```
python3 main.py -test -snapshot="./snapshot/2017-02-11_15-50-53/best_steps1500.pt"
```
The snapshot option means where your model load from. If you don't assign it, the model will start from scratch.

## Predict
* **Example1**

	```
	python3 main.py -predict="Hello my dear , I love you so much ." \
	                -snapshot="./snapshot/2017-02-11_15-50-53/best_steps1500.pt"
	```
	You will get:

	```
	Loading model from ./snapshot/2017-02-11_15-50-53/best_steps1500.pt...

	[Text]  Hello my dear , I love you so much .
	[Label] positive
	```
* **Example2**

	```
	python3 main.py -predict="You just make me so sad and I have to leave you ."\
	                -snapshot="./snapshot/2017-02-11_15-50-53/best_steps1500.pt"
	```
	You will get:

	```
	Loading model from ./snapshot/2017-02-11_15-50-53/best_steps1500.pt...

	[Text]  You just make me so sad and I have to leave you .
	[Label] negative
	```

Your text must be separated by space, even punctuation. And your text should be longer than the max kernel size.

## Changes from Original
* Removed dependency on deprecated `torchtext.data.Field`, `BucketIterator`, `Example` APIs
* Replaced with standard PyTorch `Dataset`, `DataLoader`, and custom `BucketSampler`
* Fixed `feature.data.t_()` / `target.data.sub_(1)` RuntimeError on modern PyTorch
* Fixed `loss.data[0]` IndexError (use `loss.item()` instead)
* Fixed `size_average` deprecation (use `reduction='sum'`)
* Fixed early stopping (now actually stops training)
* Added L2 weight constraint on fc layer (matching Kim's original implementation)
* Added SST dataset support with phrase-level training data
* Added Adadelta optimizer option (`-optimizer adadelta`)
* Embedding initialization with uniform[-0.25, 0.25] (matching Kim's paper)
* Added `torch.no_grad()` context for evaluation and prediction
* Compatible with PyTorch 2.0+

## Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

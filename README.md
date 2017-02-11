## Introduction
This is the implementation of kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.

Kim's implementation of the model in Theano:
[https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)

Denny Britz has an implementation in Tensorflow:
[https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy

## Usage
```
./main.py -h
```
or 

```
python3 main.py -h
```

You will get:

```
CNN text classificer

optional arguments:
  -h, --help            show this help message and exit
  -batch-size N         batch size for training [default: 50]
  -lr LR                initial learning rate [default: 0.01]
  -epochs N             number of epochs for train [default: 10]
  -dropout              the probability for dropout [default: 0.5]
  -max_norm MAX_NORM    l2 constraint of parameters
  -cpu                  disable the gpu
  -device DEVICE        device to use for iterate data
  -embed-dim EMBED_DIM
  -static               fix the embedding
  -kernel-sizes KERNEL_SIZES
                        Comma-separated kernel size to use for convolution
  -kernel-num KERNEL_NUM
                        number of each kind of kernel
  -class-num CLASS_NUM  number of class
  -shuffle              shuffle the data every epoch
  -num-workers NUM_WORKERS
                        how many subprocesses to use for data loading
                        [default: 0]
  -log-interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
  -test-interval TEST_INTERVAL
                        how many epochs to wait before testing
  -save-interval SAVE_INTERVAL
                        how many epochs to wait before saving
  -test                 train or test
  -snapshot SNAPSHOT    filename of model snapshot [default: None]
  -save-dir SAVE_DIR    where to save the checkpoint
```

## Train
```
./main.py
```
You will get:

```
Batch[100] - loss: 0.655424  acc: 59.3750%
Evaluation - loss: 0.672396  acc: 57.6923%(615/1066) 
```

## Predict
* **Example1**

	```
	./main.py -predict="Hello my dear , I love you so much ." -snapshot="./snapshot/2017-02-11-15-50/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11-15-50/snapshot_steps1500.pt]...
	
	[Text]  Hello my dear , I love you so much .
	[Label] positive
	```
* **Example2**

	```
	./main.py -predict="You just make me so sad and I have to leave you ." -snapshot="./snapshot/2017-02-11-15-50/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11-15-50/snapshot_steps1500.pt]...
	
	[Text]  You just make me so sad and I have to leave you .
	[Label] negative
	```



## Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)


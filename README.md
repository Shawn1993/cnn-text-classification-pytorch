## Introduction
Fork of Shawn Ng's [CNNs for Sentence Classification in PyTorch](https://github.com/Shawn1993/cnn-text-classification-pytorch), refactored as a scikit-learn classifier.

## Requirements
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy
* scikit-learn

## Known Issues
* The predict and predict_proba methods are probably not as efficient as they could be.
* The class probabilities returned by the predict_proba method are probably questionable.
* Doesn't play well with GridSearchCV if num_jobs isn't 1 (unless not using CUDA).
* Only supports pre-trained word vectors from TorchText (or no pre-trained vectors).
* The random_state parameter probably only works with integers or None.
* Features my idiosyncratic coding style.

## To Do
* Add support for cross-validation during training.
* Implement sample weights in eval scoring?

## Parameters
**lr : float, optional (default=0.01)**
  Initial learning rate.

**epochs : integer, optional (default=256)**
  Number of training epochs.

**batch_size : integer, optional (default=64)**
  Training batch size.

**test_interval : integer, optional (default=100)**
  The number of epochs to wait before testing.

**early_stop : integer, optional (default=1000)**
  The number of iterations without increased performance to reach before stopping.

**save_best : boolean, optional (default=True)**
  Keep the model with the best performance found during training.

**dropout : float, optional (default=0.5)**
  Dropout probability.

**max_norm : float, optional (default=0.0)**
  L2 constraint.

**embed_dim : integer, optional (default=128)**
  The number of embedding dimensions. Ignored if vectors is not None.

**kernel_num : integer, optional (default=100)**
  The number of each size of kernel.

**kernel_sizes : iterable of integers, optional (default=(3, 4, 5))**
  Kernel sizes to use for convolution.

**static : boolean, optional (default=False)**
  If true, fix the embedding.

**device : int, optional (default=-1)**
  Device to use for iterating data; -1 for CPU (see torch.cuda.set_device()).

**cuda : boolean, optional (default=True)**
  If true, use the GPU if available.

**activation_func : string, optional (default='relu')**
  Activation function. If 'relu' or 'tanh', uses rectified linear unit or hyperbolic tangent, respectively. Otherwise, uses no activation function (f(x) = x).

**scoring : callable or "roc_auc", optional (default=sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score))**
  Scoring method for testing model performance during fitting.

**pos_label : string, optional (default=None)**
  Positive class label for roc_auc scoring. Ignored if using a different scoring method.

**vectors : string, optional (default=None)**
  Which pretrained TorchText vectors to use (see [torchtext.vocab.pretrained_aliases](https://torchtext.readthedocs.io/en/latest/vocab.html#pretrained-aliases) for options).

**split_ratio : float, optional (default=0.9)**
  Ratio of training data used for training. The remainder will be used for validation.

**preprocessor : callable or None, optional (default=None)**
  Override default string preprocessing.

**class_weight : dict, "balanced" or None, optional (default=None)**
  Weights associated with each class (see class_weight parameter in existing scikit-learn classifiers).

**random_state : integer, optional (default=None)**
  Seed for the random number generator.

**verbose : integer, optional (default=0)**
  Controls the verbosity when fitting.

## Methods
**fit(X, y, sample_weight=None)**
Train the CNN classifier from the training set (X, y).
```
Parameters: X: list of strings
               The training input samples.

            y: list of strings
               The class labels.

            sample_weight: list of integers or floats, or None
               Sample weights. If None, samples are equally weighted.

Returns:    self : object
```

**predict(X)**
Predict class for X.
```
Parameters: X: list of strings
               The input samples.

Returns:    y: list of strings
               The predicted classes.
```

**predict_proba(X)**
Predict class probabilities for X.
```
Parameters: X: list of strings
               The input samples.

Returns:    y: list of lists for floats
               The predicted class probabilities.
```

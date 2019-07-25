import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, make_scorer
from time import time
from torch.autograd import Variable
from torchtext.data import Dataset, Example, Field, Iterator, Pipeline


class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.001, epochs=256, batch_size=64, test_interval=100,
                 early_stop=1000, save_best=True, dropout=0.5, max_norm=0.0,
                 embed_dim=128, kernel_num=100, kernel_sizes=(3, 4, 5),
                 static=False, device=-1, cuda=True, activation_func="relu",
                 scoring=make_scorer(accuracy_score), vectors=None,
                 split_ratio=0.9, preprocessor=None, class_weight=None,
                 random_state=None, verbose=0):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_interval = test_interval
        self.early_stop = early_stop
        self.save_best = save_best
        self.dropout = dropout
        self.max_norm = max_norm
        self.embed_dim = embed_dim
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.static = static
        self.device = device
        self.cuda = cuda
        self.activation_func = activation_func
        self.scoring = scoring
        self.vectors = vectors
        self.split_ratio = split_ratio
        self.preprocessor = preprocessor
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

    def __clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def __eval(self, data_iter):
        self.__model.eval()

        preds, targets = [], []

        for batch in data_iter:
            feature, target = batch.text.data.t(), batch.label.data.sub(1)

            if self.cuda and torch.cuda.is_available():
                feature, target = feature.cuda(), target.cuda()

            logit = self.__model(feature)

            F.cross_entropy(logit, target, reduction="sum")

            preds += torch.max(logit, 1)[1].view(target.size()).data.tolist()
            targets += target.data.tolist()

        preds = [self.__label_field.vocab.itos[pred + 1] for pred in preds]
        targets = [self.__label_field.vocab.itos[targ + 1] for targ in targets]
        return self.scoring(_Eval(preds), None, targets)

    def fit(self, X, y, sample_weight=None):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        torch.backends.cudnn.deterministic = self.random_state is not None
        torch.backends.cudnn.benchmark = self.random_state is None

        if self.verbose > 1:
            params = self.get_params().items()

            print("Fitting with the following parameters:")
            print("\n".join([": ".join([k, str(v)]) for k, v in params]))

        start = time() if self.verbose > 0 else None
        train_iter, dev_iter = self.__preprocess(X, y, sample_weight)
        embed_num = len(self.__text_field.vocab)
        class_num = len(self.__label_field.vocab) - 1
        self.__model = _CNNText(embed_num, self.embed_dim, class_num,
                                self.kernel_num, self.kernel_sizes,
                                self.dropout, self.static,
                                self.activation_func,
                                vectors=self.__text_field.vocab.vectors)

        if self.cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            self.__model.cuda()

        optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.lr,
                                     weight_decay=self.max_norm)
        best_model = self.__model
        steps, best_acc, last_step = 0, 0, 0
        active = True

        self.__model.train()

        for epoch in range(self.epochs):
            for batch in train_iter:
                feature, target = batch.text.data.t(), batch.label.data.sub(1)

                if self.cuda and torch.cuda.is_available():
                    feature, target = feature.cuda(), target.cuda()

                optimizer.zero_grad()
                F.cross_entropy(self.__model(feature), target).backward()
                optimizer.step()

                steps += 1

                if steps % self.test_interval == 0:
                    dev_acc = self.__eval(dev_iter)

                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        last_step = steps

                        if self.save_best:
                            best_model = deepcopy(self.__model)
                    elif steps - last_step >= self.early_stop:
                        active = False
                        break

            if not active:
                break

        self.__model = best_model if self.save_best else self.__model
        self.classes_ = self.__label_field.vocab.itos[1:]

        if self.verbose > 0:
            self.__print_elapsed_time(time() - start)

        torch.cuda.empty_cache()
        return self

    def __predict(self, X):
        y_output = []
        max_kernel_size = max(self.kernel_sizes)

        self.__model.eval()

        for text in X:
            assert isinstance(text, str)

            text = self.__text_field.preprocess(text)
            text = self.__pad(text, max_kernel_size, True)
            text = [[self.__text_field.vocab.stoi[x] for x in text]]
            x = Variable(torch.tensor(text))
            x = x.cuda() if self.cuda and torch.cuda.is_available() else x

            y_output.append(self.__model(x))

        torch.cuda.empty_cache()
        return y_output

    def predict(self, X):
        y_pred = [torch.argmax(yi, 1) for yi in self.__predict(X)]
        return [self.__label_field.vocab.itos[yi.data[0] + 1] for yi in y_pred]

    def predict_proba(self, X):
        softmax = nn.Softmax(dim=1)
        y_prob = [softmax(yi) for yi in self.__predict(X)]
        return [[float(yij) for yij in yi[0]] for yi in y_prob]

    def __pad(self, x, max_kernel_size, preprocessed=False):
        tokens = x if preprocessed else self.__text_field.preprocess(x)
        difference = max_kernel_size - len(tokens)

        if difference > 0:
            padding = [self.__text_field.pad_token] * difference
            return x + padding if preprocessed else " ".join([x] + padding)

        return x

    def __preprocess(self, X, y, sample_weight):
        self.__text_field = Field(lower=True)
        self.__label_field = Field(sequential=False)
        self.__text_field.preprocessing = Pipeline(self.__preprocess_text)
        max_kernel_size = max(self.kernel_sizes)
        sample_weight = None if sample_weight is None else list(sample_weight)

        for i in range(len(X)):
            X[i] = self.__pad(X[i], max_kernel_size)

        fields = [("text", self.__text_field), ("label", self.__label_field)]
        exmpl = [Example.fromlist([X[i], y[i]], fields) for i in range(len(X))]
        weights = [1 for yi in y] if sample_weight is None else sample_weight

        if self.class_weight is not None:
            cw = self.class_weight

            if isinstance(cw, str) and cw == "balanced":
                counter = Counter(y)
                cw = [len(y) / (len(counter) * counter[yi]) for yi in y]
                weights = [weights[i] * cw[i] for i in range(len(y))]
            elif isinstance(cw, dict):
                cw = [cw[yi] for yi in y]
                weights = [weights[i] * cw[i] for i in range(len(y))]

        min_weight = min(weights)
        weights = [round(w / min_weight) for w in weights]

        for i in range(len(X)):
            if weights[i] > 1:
                Xi = [X[i] for j in range(weights[i] - 1)]
                exmpl += [Example.fromlist([x, y[i]], fields) for x in Xi]

        train_data, dev_data = Dataset(exmpl, fields).split(self.split_ratio,
                                                            self.random_state,)

        self.__text_field.build_vocab(train_data, dev_data,
                                      vectors=self.vectors)
        self.__label_field.build_vocab(train_data, dev_data)

        batch_sizes = (self.batch_size, len(dev_data))
        return Iterator.splits((train_data, dev_data), batch_sizes=batch_sizes,
                               sort_key=lambda ex: len(ex.text), repeat=False)

    def __preprocess_text(self, text):
        if self.preprocessor is None:
            return self.__clean_str(text)

        return self.__clean_str(self.preprocessor(text))

    def __print_elapsed_time(self, seconds):
        sc = round(seconds)
        mn = int(sc / 60)
        sc = sc % 60
        hr = int(mn / 60)
        mn = mn % 60
        hr = "{} hour{}".format(hr, "s" if hr > 1 else "") if hr > 0 else ""
        mn = "{} minute{}".format(mn, "s" if mn > 1 else "") if mn > 0 else ""
        sc = "{} second{}".format(sc, "s" if sc > 1 else "") if sc > 0 else ""
        times = [t for t in [hr, mn, sc] if len(t) > 0]

        if len(times) == 3:
            times = " and ".join([", ".join([hr, mn]), sc])
        elif len(times) == 2:
            times = " and ".join(times)
        else:
            times = times[0] if len(times) > 0 else "less than 1 second"

        print("Completed training in {}.".format(times))


class _CNNText(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num,
                 kernel_sizes, dropout, static, activation_func, vectors=None):
        super(_CNNText, self).__init__()

        if vectors is None:
            self.__embed = nn.Embedding(embed_num, embed_dim)
        else:
            self.__embed = nn.Embedding.from_pretrained(vectors)
            embed_dim = self.__embed.embedding_dim

        Ks = kernel_sizes
        module_list = [nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in Ks]
        self.__convs1 = nn.ModuleList(module_list)
        self.__dropout = nn.Dropout(dropout)
        self.__fc1 = nn.Linear(len(Ks) * kernel_num, class_num)
        self.__static = static

        if activation_func == "relu":
            self.__f = F.relu
        elif activation_func == "tanh":
            self.__f = torch.tanh
        else:
            self.__f = lambda x: x

    def forward(self, x):
        x = Variable(self.__embed(x)) if self.__static else self.__embed(x)
        x = [self.__f(cnv(x.unsqueeze(1))).squeeze(3) for cnv in self.__convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        return self.__fc1(self.__dropout(torch.cat(x, 1)))

class _Eval():
    def __init__(self, preds):
        self.__preds = preds

    def predict(self, X):
        return self.__preds

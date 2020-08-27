import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from os import remove
from os.path import exists
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split as split
from sklearn.utils.class_weight import compute_sample_weight
from time import time
from torchtext.data import Dataset, Example, Field, Iterator


class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.001, epochs=256, batch_size=64, test_interval=100,
                 early_stop=1000, save_best=True, dropout=0.5, max_norm=0.0,
                 embed_dim=128, kernel_num=100, kernel_sizes=(3, 4, 5),
                 static=False, device=-1, cuda=True, activation_func="relu",
                 scoring=make_scorer(accuracy_score), pos_label=None,
                 vectors=None, split_ratio=0.8, preprocessor=None,
                 class_weight=None, random_state=None, verbose=0):
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
        self.pos_label = pos_label
        self.vectors = vectors
        self.split_ratio = split_ratio
        self.preprocessor = preprocessor
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose
        self.__max_kernel_size = max(self.kernel_sizes)

    def __default_preprocessor(self, string):
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
        softmax = nn.Softmax(dim=1) if self.scoring == "roc_auc" else None

        for batch in data_iter:
            feature, target = batch.text.t_(), batch.label.sub_(1)

            if self.cuda and torch.cuda.is_available():
                feature, target = feature.cuda(), target.cuda()

            logit = self.__model(feature)

            F.cross_entropy(logit, target, reduction="sum")

            if self.scoring == "roc_auc":
                pred = [[float(p) for p in dist] for dist in softmax(logit)]
            else:
                pred = torch.max(logit, 1)[1].view(target.size()).tolist()

            preds += pred
            targets += target.tolist()

        targets = [self.__label_field.vocab.itos[targ + 1] for targ in targets]

        if self.scoring == "roc_auc":
            pos_index = self.__label_field.vocab.stoi[self.pos_label] - 1
            return roc_auc_score(targets, [pred[pos_index] for pred in preds])

        preds = [self.__label_field.vocab.itos[pred + 1] for pred in preds]
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
        train_iter, dev_iter = self.__prepare_train_data(X, y, sample_weight)
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
        steps, best_acc, last_step = 0, 0, 0
        active = True
        filename = "./{}.model".format(time())

        self.__model.train()

        for epoch in range(self.epochs):
            for batch in train_iter:
                feature, target = batch.text.t_(), batch.label.sub_(1)

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
                            torch.save(self.__model.state_dict(), filename)
                    elif steps - last_step >= self.early_stop:
                        active = False
                        break

            if not active:
                break

        if self.save_best and exists(filename):
            self.__model.load_state_dict(torch.load(filename))
            remove(filename)

        self.classes_ = self.__label_field.vocab.itos[1:]

        if self.verbose > 0:
            self.__print_elapsed_time(time() - start)

        return self

    def __predict(self, X):
        texts = []

        self.__model.eval()

        for text in X:
            assert isinstance(text, str)

            text = self.__text_field.preprocess(text)
            text = [self.__text_field.vocab.stoi[x] for x in text]
            texts.append(torch.tensor(text))

        x = torch.stack(texts, 0)
        x = x.cuda() if self.cuda and torch.cuda.is_available() else x
        return self.__model(x)

    def predict(self, X):
        y_pred = torch.argmax(self.__predict(X), 1)
        return [self.__label_field.vocab.itos[yi.item() + 1] for yi in y_pred]

    def predict_proba(self, X):
        return nn.Softmax(dim=1)(self.__predict(X)).tolist()

    def __prepare_train_data(self, X, y, sample_weight):
        self.__text_field = Field(lower=True)
        self.__label_field = Field(sequential=False)
        self.__text_field.tokenize = self.__tokenize
        sample_weight = None if sample_weight is None else list(sample_weight)
        sw = [1 for yi in y] if sample_weight is None else sample_weight
        s = y if Counter(y).most_common()[-1][1] > 1 else None
        X_t, X_d, y_t, y_d, w_t, _ = split(X, y, sw, shuffle=True, stratify=s,
                                           random_state=self.random_state,
                                           train_size=self.split_ratio)
        fields = [("text", self.__text_field), ("label", self.__label_field)]
        examples = [[X_t[i], y_t[i]] for i in range(len(X_t))]
        examples = [Example.fromlist(example, fields) for example in examples]
        weights = compute_sample_weight(self.class_weight, y_t)
        weights = [weights[i] * w_t[i] for i in range(len(y_t))]
        min_weight = min(weights)
        weights = [int(round(weight / min_weight)) for weight in weights]

        for i in range(len(X_t)):
            Xi = [X_t[i] for j in range(weights[i] - 1)]
            examples += [Example.fromlist([x, y_t[i]], fields) for x in Xi]

        train_data = Dataset(examples, fields)
        dev_data = [[X_d[i], y_d[i]] for i in range(len(X_d))]
        dev_data = [Example.fromlist(example, fields) for example in dev_data]
        dev_data = Dataset(dev_data, fields)

        self.__text_field.build_vocab(train_data, dev_data,
                                      vectors=self.vectors)
        self.__label_field.build_vocab(train_data, dev_data)

        batch_sizes = (self.batch_size, len(dev_data))
        return Iterator.splits((train_data, dev_data), batch_sizes=batch_sizes,
                               sort_key=lambda ex: len(ex.text), repeat=False)

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

    def __tokenize(self, text):
        if self.preprocessor is None:
            text = self.__default_preprocessor(text)
        else:
            text = self.preprocessor(text)

        tokens = text.split()
        difference = self.__max_kernel_size - len(tokens)
        return tokens + [self.__text_field.pad_token] * max(difference, 0)


class _CNNText(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num,
                 kernel_sizes, dropout, static, activation_func, vectors=None):
        super(_CNNText, self).__init__()

        if vectors is None:
            self.__embed = nn.Embedding(embed_num, embed_dim)
            self.__embed.weight.requires_grad = not static
        else:
            self.__embed = nn.Embedding.from_pretrained(vectors, freeze=static)
            embed_dim = self.__embed.embedding_dim

        Ks = kernel_sizes
        module_list = [nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in Ks]
        self.__convs = nn.ModuleList(module_list)
        self.__dropout = nn.Dropout(dropout)
        self.__fc = nn.Linear(len(Ks) * kernel_num, class_num)

        if activation_func == "relu":
            self.__f = F.relu
        elif activation_func == "tanh":
            self.__f = torch.tanh
        else:
            self.__f = lambda x: x

    def forward(self, x):
        x = self.__embed(x).unsqueeze(1)
        x = [self.__f(cnv(x), inplace=True).squeeze(3) for cnv in self.__convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        return self.__fc(self.__dropout(torch.cat(x, 1)))


class _Eval():
    def __init__(self, preds):
        self.__preds = preds

    def predict(self, X):
        return self.__preds

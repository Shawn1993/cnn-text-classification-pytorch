import re
import os
import random
import tarfile
import urllib
from collections import Counter
import torch
from torch.utils.data import Dataset, Sampler


class Vocab:
    """Simple vocabulary class to replace legacy torchtext Field.vocab."""

    def __init__(self, counter, specials=('<pad>', '<unk>')):
        self.itos = list(specials)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        # Sort by frequency desc, then alphabetically for deterministic ordering
        for word, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def __len__(self):
        return len(self.itos)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class MRDataset(Dataset):
    """Movie Review polarity dataset using standard PyTorch Dataset."""

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar.gz'
    dirname = 'rt-polaritydata'

    def __init__(self, examples, text_vocab, label_vocab):
        self.examples = examples  # list of (text_tokens, label_str)
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, label = self.examples[idx]
        text_ids = [self.text_vocab.stoi.get(t, 1) for t in tokens]  # 1 = <unk>
        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        label_id = self.label_vocab.stoi[label]
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        return text_tensor, label_tensor

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                for member in tfile.getmembers():
                    member_path = os.path.join(root, member.name)
                    if not is_within_directory(root, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tfile.extractall(root)
        return os.path.join(path, '')

    @classmethod
    def splits(cls, dev_ratio=0.1, shuffle=True, root='.'):
        """Create train/dev dataset splits with vocabularies.

        Returns:
            train_dataset, dev_dataset, text_vocab, label_vocab
        """
        path = cls.download_or_unzip(root)

        examples = []
        with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = clean_str(line).lower().split()
                    examples.append((tokens, 'negative'))
        with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = clean_str(line).lower().split()
                    examples.append((tokens, 'positive'))

        if shuffle:
            random.shuffle(examples)

        # Build vocabularies from all data
        text_counter = Counter()
        label_counter = Counter()
        for tokens, label in examples:
            text_counter.update(tokens)
            label_counter[label] += 1

        text_vocab = Vocab(text_counter)
        label_vocab = Vocab(label_counter, specials=[])

        # Split into train/dev
        dev_index = -1 * int(dev_ratio * len(examples))
        train_examples = examples[:dev_index]
        dev_examples = examples[dev_index:]

        train_dataset = cls(train_examples, text_vocab, label_vocab)
        dev_dataset = cls(dev_examples, text_vocab, label_vocab)

        return train_dataset, dev_dataset, text_vocab, label_vocab


class SSTDataset(Dataset):
    """Stanford Sentiment Treebank dataset (SST-1, 5-class fine-grained)."""

    # Preprocessed SST data from Harvard NLP (sent-conv-torch)
    base_url = 'https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/'
    filenames = {
        'train': 'stsa.fine.train',
        'phrases_train': 'stsa.fine.phrases.train',
        'dev': 'stsa.fine.dev',
        'test': 'stsa.fine.test',
    }

    def __init__(self, examples, text_vocab, label_vocab):
        self.examples = examples  # list of (text_tokens, label_str)
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, label = self.examples[idx]
        text_ids = [self.text_vocab.stoi.get(t, 1) for t in tokens]  # 1 = <unk>
        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        label_id = self.label_vocab.stoi[label]
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        return text_tensor, label_tensor

    @classmethod
    def download_if_needed(cls, root):
        """Download SST data files if not present."""
        data_dir = os.path.join(root, 'sst')
        os.makedirs(data_dir, exist_ok=True)
        for split, fname in cls.filenames.items():
            fpath = os.path.join(data_dir, fname)
            if not os.path.isfile(fpath):
                url = cls.base_url + fname
                print(f'Downloading {fname}...')
                urllib.request.urlretrieve(url, fpath)
        return data_dir

    @classmethod
    def _read_file(cls, filepath):
        """Read SST format: each line is 'label text'."""
        examples = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # SST format: "label text..." where label is a single digit 0-4
                label = line[0]
                text = line[2:]  # skip "label "
                # SST is already tokenized, no need for clean_str
                tokens = text.lower().split()
                examples.append((tokens, label))
        return examples

    @classmethod
    def splits(cls, root='.', use_phrases=True):
        """Create train/dev/test dataset splits with vocabularies.

        Args:
            root: data directory root
            use_phrases: if True, use phrase-level training data (as in Kim's paper)

        Returns:
            train_dataset, dev_dataset, test_dataset, text_vocab, label_vocab
        """
        data_dir = cls.download_if_needed(root)

        if use_phrases:
            train_examples = cls._read_file(
                os.path.join(data_dir, cls.filenames['phrases_train']))
        else:
            train_examples = cls._read_file(
                os.path.join(data_dir, cls.filenames['train']))
        dev_examples = cls._read_file(os.path.join(data_dir, cls.filenames['dev']))
        test_examples = cls._read_file(os.path.join(data_dir, cls.filenames['test']))

        # Build vocabularies from all data
        text_counter = Counter()
        label_counter = Counter()
        for tokens, label in train_examples + dev_examples + test_examples:
            text_counter.update(tokens)
            label_counter[label] += 1

        text_vocab = Vocab(text_counter)
        label_vocab = Vocab(label_counter, specials=[])

        train_dataset = cls(train_examples, text_vocab, label_vocab)
        dev_dataset = cls(dev_examples, text_vocab, label_vocab)
        test_dataset = cls(test_examples, text_vocab, label_vocab)

        return train_dataset, dev_dataset, test_dataset, text_vocab, label_vocab


class BucketSampler(Sampler):
    """Sort samples by text length within buckets to minimize padding."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Get lengths
        lengths = [len(self.dataset.examples[i][0]) for i in range(len(self.dataset))]
        indices = list(range(len(self.dataset)))

        # Sort by length
        indices = sorted(indices, key=lambda i: lengths[i])

        # Create batches of similar-length sequences
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        # Shuffle batch order (but keep within-batch sorting)
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch, min_len=5):
    """Pad sequences in a batch to the same length (at least min_len for conv kernels)."""
    texts, labels = zip(*batch)
    max_len = max(max(len(t) for t in texts), min_len)
    padded = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i, :len(t)] = t
    labels = torch.stack(labels)
    return padded, labels

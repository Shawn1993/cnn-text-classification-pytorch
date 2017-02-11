import re
import os
import random
from torchtext import data


class MR(data.ZipDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = os.path.join('rt-polaritydata', 'rt-polaritydata')

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path='.', examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
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

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = os.path.expanduser(path)
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg')) as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'rt-polarity.pos')) as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))

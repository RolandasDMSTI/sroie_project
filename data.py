from collections import Counter
import string

import torch
from torch.utils.data import Dataset, DataLoader

def token_features(token):
    """
    Return categorical features for one token.

    Features:
        0: capitalization
        1: length bucket
        2: contains digit
        3: all digits
        4: contains hyphen
        5: contains punctuation
    """

    # Capitalization
    if token.islower():
        capitalization = 0
    elif token.isupper():
        capitalization = 1
    elif token.istitle():
        capitalization = 2
    elif any(c.isupper() for c in token):
        capitalization = 3
    else:
        capitalization = 4

    # Word length
    length = len(token)

    if length <= 2:
        length_bucket = 0
    elif length <= 4:
        length_bucket = 1
    elif length <= 7:
        length_bucket = 2
    elif length <= 12:
        length_bucket = 3
    else:
        length_bucket = 4

    return [
        capitalization,
        length_bucket,
        int(any(c.isdigit() for c in token)),
        int(token.isdigit()),
        int("-" in token),
        int(any(c in string.punctuation for c in token)),
    ]


PAD = "<PAD>"
UNK = "<UNK>"

class Vocabulary:

    def __init__(self, sequences, min_freq=1):
        counts = Counter(token for sequence in sequences for token in sequence)

        self.token2id = {PAD: 0, UNK: 1}

        for token, count in counts.items():
            if count >= min_freq:
                self.token2id[token] = len(self.token2id)

    def __len__(self):
        return len(self.token2id)

    def encode(self, sequence):
        return [self.token2id.get(token, self.token2id[UNK]) for token in sequence]

from statistics import median

class NERDataset(Dataset):

    def __init__(self, data, vocab):
        self.sentences = [vocab.encode(x["tokens"]) for x in data]
        self.features = [[token_features(token) for token in x["tokens"]] for x in data]
        self.sizes = [[size / median(x["sizes"]) - 1 for size in x["sizes"]] for x in data]
        self.labels = [x["ner_tags"] for x in data]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i], self.features[i], self.sizes[i], self.labels[i]


def collate_fn(batch):
    sentences, features, sizes, labels = zip(*batch)
    max_len = max(map(len, sentences))
    num_features = len(features[0][0])
    assert num_features == 6

    ID_O = 0 #ID for 'O'
    input_ids = [s + [0] * (max_len - len(s)) for s in sentences]
    tags = [y + [ID_O] * (max_len - len(y)) for y in labels]
    padded_features = [f + [[0] * num_features] * (max_len - len(f)) for f in features]
    padded_sizes = [s + [0.0] * (max_len - len(s)) for s in sizes]
    mask = [[True] * len(s) + [False] * (max_len - len(s)) for s in sentences]

    return {
        "input_ids": torch.tensor(input_ids),
        "features": torch.tensor(padded_features),
        "sizes": torch.tensor(padded_sizes, dtype=torch.float),
        "tags": torch.tensor(tags),
        "mask": torch.tensor(mask),
    }

def make_dataloader(data, vocab, batch_size=32, shuffle=True):
    dataset = NERDataset(data, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def create_loaders(dataset, batch_size):
    train_sentences = [x["tokens"] for x in dataset["train"]]
    vocab = Vocabulary(train_sentences)
    train_loader = make_dataloader(dataset["train"], vocab, batch_size=batch_size)
    val_loader = make_dataloader(dataset["validation"], vocab, batch_size=batch_size, shuffle=False)
    test_loader = make_dataloader(dataset["test"], vocab, batch_size=batch_size, shuffle=False)
    return vocab, train_loader, val_loader, test_loader

# Word2Vec
def load_word2vec(model, vocab):
    embedding_dim = model.vector_size
    matrix = torch.randn(len(vocab), embedding_dim) * 0.05

    matrix[vocab.token2id["<PAD>"]] = 0

    for word, idx in vocab.token2id.items():
        if word in model:
            matrix[idx] = torch.tensor(model[word])

    return matrix

# GloVe
def load_glove(path, vocab, embedding_dim):
    matrix = torch.randn(len(vocab), embedding_dim) * 0.05
    matrix[vocab.token2id[PAD]] = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]

            if word not in vocab.token2id:
                continue

            vector = torch.tensor([float(x) for x in values[1:]])
            matrix[vocab.token2id[word]] = vector

    return matrix

# FastText
def load_fasttext(path, vocab, embedding_dim):
    matrix = torch.randn(len(vocab), embedding_dim) * 0.05
    matrix[vocab.token2id["<PAD>"]] = 0

    with open(path, encoding="utf-8") as f:
        first_line = f.readline()

        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]

            if word not in vocab.token2id:
                continue

            vector = torch.tensor([float(x) for x in values[1:]])
            matrix[vocab.token2id[word]] = vector

    return matrix


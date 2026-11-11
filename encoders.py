import torch.nn as nn
from gensim.models import KeyedVectors
from data import load_word2vec, load_glove, load_fasttext

class LearnableEncoder(nn.Module):
    """Train word embeddings from scratch."""

    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.output_dim = embedding_dim

    def forward(self, input_ids, attention_mask=None):
        return self.embedding(input_ids)


class Word2VecEncoder(nn.Module):

    def __init__(self, embedding_matrix, padding_idx=0, freeze=False):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=padding_idx)
        self.output_dim = embedding_matrix.shape[1]

    def forward(self, input_ids, attention_mask=None):
        return self.embedding(input_ids)


class GloVeEncoder(nn.Module):

    def __init__(self, embedding_matrix, padding_idx=0, freeze=False):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze,
            padding_idx=padding_idx
        )
        self.output_dim = embedding_matrix.shape[1]

    def forward(self, input_ids, attention_mask=None):
        return self.embedding(input_ids)


class FastTextEncoder(nn.Module):

    def __init__(self, embedding_matrix, padding_idx=0, freeze=False):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze,
            padding_idx=padding_idx
        )
        self.output_dim = embedding_matrix.shape[1]

    def forward(self, input_ids, attention_mask=None):
        return self.embedding(input_ids)


class BERTEncoder(nn.Module):
    """Use a pretrained BERT model as the encoder."""

    def __init__(self, bert_model):
        super().__init__()

        self.bert = bert_model
        self.output_dim = bert_model.config.hidden_size

    def forward(self, input_ids, attention_mask=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state

def create_encoder(vocab, encoder="learnable", freeze=False):

    match encoder:
        case "learnable":
            encoder = LearnableEncoder(vocab_size=len(vocab), embedding_dim=100)
        case "word2vec": # not tested
            # https://radimrehurek.com/gensim/models/word2vec.html
            # https://www.nltk.org/howto/gensim.html
            # https://code.google.com/archive/p/word2vec/
            word2vec = KeyedVectors.load_word2vec_format("word2vec.bin", binary=True)
            word2vec_matrix = load_word2vec(word2vec, vocab)
            encoder = Word2VecEncoder(word2vec_matrix, freeze=freeze)
        case "glove": # not tested
            glove_matrix = load_glove("glove.6B.100d.txt", vocab, 100)
            encoder = GloVeEncoder(glove_matrix, freeze=freeze)
        case "fasttext": # not tested
            # https://fasttext.cc/docs/en/crawl-vectors.html
            # https://fasttext.cc/docs/en/english-vectors.html
            # fasttext_matrix = load_fasttext("cc.en.300.vec", vocab, 300)
            fasttext_matrix = load_fasttext("crawl-300d-2M.vec", vocab, 300)
            encoder = FastTextEncoder(fasttext_matrix, freeze=freeze)
        case "bert": # not implemented yet
            # encoder = BERTEncoder(bert_model)
            raise NotImplementedError("BERT not implemented yet")
    return encoder

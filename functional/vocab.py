import joblib
from data.config import image_captioning
import os

class Vocabulary_nltk:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    UNK_token = 3

    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": self.PAD_token, "<SOS>": self.SOS_token, "<EOS>": self.EOS_token,
                           "<UNK>": self.UNK_token}
        self.word2count = {}
        self.index2word = {self.PAD_token: "<PAD>", self.SOS_token: "<SOS>", self.EOS_token: "<EOS>",
                           self.UNK_token: "<UNK>"}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0

    def get_pad(self):
        return self.word2index["<PAD>"]

    def get_sos(self):
        return self.word2index["<SOS>"]

    def get_eos(self):
        return self.word2index["<EOS>"]

    def get_unk(self):
        return self.word2index["<UNK>"]

    def has_word(self, word) -> bool:
        return word in self.word2index

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        return self.get_unk()

    def get_word(self, index):
        return self.index2word[index]

    def size(self):
        return len(self.index2word)

    def is_empty(self):
        empty_size = 4
        return self.size() <= empty_size

    def short_dict(self, freq):
        cache_del = []
        cache_reindex = []
        for key, value in self.word2count.items():
            if value < freq:
                cache_del.append(key)
            else:
                cache_reindex.append(key)
        for key in cache_del:
            del self.word2count[key]
        self.reset(with_count=False)
        for key in cache_reindex:
            self.word2index[key] = self.num_words
            self.index2word[self.num_words] = key
            self.num_words += 1

    def reset(self, with_count=True):
        if with_count:
            self.word2count = {}
            self.num_sentences = 0
            self.longest_sentence = 0
        self.word2index = {"<PAD>": self.PAD_token, "<SOS>": self.SOS_token, "<EOS>": self.EOS_token,
                           "<UNK>": self.UNK_token}
        self.index2word = {self.PAD_token: "<PAD>", self.SOS_token: "<SOS>", self.EOS_token: "<EOS>",
                           self.UNK_token: "<UNK>"}
        self.num_words = 4

    def vocab_save(self, path):
        vocab_database = {
            'word2index': self.word2index,
            'word2count': self.word2count,
            'index2word': self.index2word,
            'num_words': self.num_words,
            'num_sentences': self.num_sentences,
            'longest_sentence': self.longest_sentence
        }
        joblib.dump(vocab_database, __file__[:-len('vocab.py')] + path)

    def vocab_load(self, path):
        vocab_database = joblib.load(__file__[:-len('vocab.py')] + path)
        self.reset()
        self.word2index = vocab_database['word2index']
        self.word2count = vocab_database['word2count']
        self.index2word = vocab_database['index2word']
        self.num_words = vocab_database['num_words']
        self.num_sentences = vocab_database['num_sentences']
        self.longest_sentence = vocab_database['longest_sentence']
        del vocab_database

# To upload early saved vocab
vocabulary = Vocabulary_nltk('Uploaded')
vocabulary.vocab_load(path=image_captioning['paths']['VOCAB_PATH'])
VOCAB_SIZE = vocabulary.size()
PAD_IDX = vocabulary.get_pad()
EOS_IDX = vocabulary.get_eos()
SOS_IDX = vocabulary.get_sos()
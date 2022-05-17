import pickle
from collections import Counter


class Vocab:
    def __init__(self, corpus_pth, min_freq=1, special_tokens=["<PAD>", "SEP", "<MASK>", "<UNK>", "CLS"], pth=None):
        self.corpus_pth = corpus_pth
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        self.pth = pth
        self.split_token = "<SEP>"
        self.unk_token = "<UNK>"
        self.first_token = "<CLS>"
        self.pad_token = "<PAD>"
        self.word_to_ix = dict()
        self.id_to_word = dict()

    def build_vocab(self):
        counter = Counter()
        f = open(self.corpus_pth, "r", encoding="utf-8")
        for i in f:
            word_lst = i.strip().split()
            for word in word_lst:
                counter[word] += 1
        freq_tupe = sorted(counter.items(), key=lambda key: key[1])
        del counter

        for ix, token in enumerate(self.special_tokens):
            self.word_to_ix[token] = ix
        shift = len(self.special_tokens)
        for ix, (word, freq) in enumerate(freq_tupe):
            self.word_to_ix[word] = (ix + shift)
            if freq < self.min_freq:
                break
        for k, v in self.word_to_ix.items():
            self.id_to_word[v] = k

    @staticmethod
    def load_vocab(pth):
        with open(pth, "rb") as f:
            return pickle.load(f)

    def save_vocab(self):
        with open(self.pth, "wb") as f:
            pickle.dump(self, f)


class Sentence:
    def __init__(self, sentence, vocab_obj, max_len=100):
        self.sentence = sentence
        self.tokens = []
        self.word_to_ix = vocab_obj.word_to_ix
        self.id_to_word = vocab_obj.id_to_word
        self.split_token = vocab_obj.split_token
        self.unk_token = vocab_obj.unk_token
        self.pad_token = vocab_obj.pad_token
        self.max_len = max_len

    def to_tokens(self):
        for word in self.sentence.split():
            if len(self.tokens) > self.max_len:
                break
            if word == self.split_word:
                self.tokens.append(self.word_to_ix[self.split_word])
                continue
            if word not in self.word_to_ix:
                self.tokens.append(self.word_to_ix[self.unk_token])
                continue
            self.tokens.append(self.word_to_ix[word])

        if len(self.tokens) < self.max_len:
            for _ in range(self.max_len - len(self.tokens)):
                self.tokens.append(self.word_to_ix[self.pad_token])

    def from_tokens(self):
        sentence = ""
        for token in self.tokens:
            sentence += self.id_to_word[token]
        return sentence


class Document:
    def __init__(self):
        pass

if __name__ == '__main__':
    corpus_path = r"data/corpus.small"

    vocab = Vocab(corpus_path, min_freq=1)
    vocab.build_vocab()
    # vocab.save_vocab()
    print(vocab.word_to_ix)
    print(vocab.id_to_word)
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i in f:
            sen = i.strip()
            print(sen)
            sentence = Sentence(sen, vocab)
            sentence.to_tokens()
            print(sentence.tokens)





#     corpus_path = r"data/corpus.small"
#     output_path = r"data/vocab.small"
#     encoding = "utf-8"
#     min_freq = 1
#     with open(corpus_path, "r", encoding=encoding) as f:
#         vocab = WordVocab(f, min_freq=min_freq)
#     vocab.save_vocab(output_path)
























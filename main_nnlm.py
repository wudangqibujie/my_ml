import torch
from collections import Counter
import random
import torch.nn as nn
import torch.optim as optim


class Corpus:
    def __init__(self, raw_text_paths, output_file_signature):
        self.raw_text_paths = raw_text_paths
        self.output_file_signature = output_file_signature

    def write_corpus(self):
        pass


class Vocab:
    def __init__(self, corpus_pths, min_count=1):
        self.corpus_pths = corpus_pths
        self.counter = Counter()
        self.min_count = min_count
        self.vocab = []

    def _count_word(self, pth):
        f = open(pth, encoding="utf-8")
        for i in f:
            i = i.strip()
            for word in i.split():
                self.counter[word] += 1
        f.close()

    def cal_totals(self):
        for fil in self.corpus_pths:
            self._count_word(fil)

    def filter_min_freq(self):
        cnt_map = self.counter.items()
        rs = sorted(cnt_map, key=lambda x: x[1], reverse=True)
        for i in rs:
            if i[1] <= self.min_count:
                break
            self.vocab.append(i[0])

    def write_vocab(self):
        with open("vocab_jayliu.txt" , "w", encoding="utf-8") as f:
            for v in self.vocab:
                f.write(v + "\n")


class Tokenizer:
    def __init__(self, vocab_pth):
        self.word_2_ix = dict()
        self.unk_token = "<UNK>"
        self.vocab = self._load_vocab(vocab_pth)

    def _load_vocab(self, pth):
        f = open(pth, encoding="utf-8")
        self.word_2_ix[self.unk_token] = 0
        gap = 1
        for ix, i in enumerate(f):
            self.word_2_ix[i.strip()] = ix + gap

    def trans_to_tokens(self, sen):
        tokens = []
        sen_lst = sen.split()
        for s in sen_lst:
            if s not in self.word_2_ix:
                tokens.append(self.word_2_ix[self.unk_token])
                continue
            tokens.append(self.word_2_ix[s])
        return tokens


class Dataset:
    def __init__(self, train_pths, tokenizer, shuffle=False, batch_size=64, win_size=5):
        self.tokenizer = tokenizer
        self.train_pths = train_pths if not shuffle else random.shuffle(train_pths)
        self.batch_size = batch_size
        self.win_size = win_size

    def get_batch(self):
        batch_data = []
        for file_ in self.train_pths:
            f = open(file_, encoding="utf-8")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # print(line)
                tokens = self.tokenizer.trans_to_tokens(line)
                # print(tokens)
                silce_tokens = self.windowise(tokens)
                for sample in silce_tokens:
                    target = sample[-1]
                    input_ = sample[: -1]
                    batch_data.append((input_, target))
                    if len(batch_data) >= self.batch_size:
                        yield batch_data
                        batch_data = []

    def windowise(self, tokens):
        if len(tokens) < self.win_size:
            return []
        rs = []
        for ix in range(self.win_size, len(tokens)):
            rs.append(tokens[ix - self.win_size: ix])
        return rs


class NNLM(nn.Module):
    def __init__(self, word_num, win_size, hidden_size=64, word_dim=64):
        self.word_dim = word_dim
        self.win_size = win_size
        super(NNLM, self).__init__()
        self.word_matrix = nn.Embedding(word_num, word_dim)
        self.liner1 = nn.Linear((self.win_size - 1) * self.word_dim, hidden_size, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(hidden_size))
        self.activ1 = nn.Tanh()
        self.liner2 = nn.Linear(hidden_size, word_num)
        self.bias2 = nn.Parameter(torch.zeros(word_num))

    def forward(self, X):
        out = self.word_matrix(X)
        out = out.view(-1, (self.win_size - 1) * self.word_dim)
        out = self.activ1(self.liner1(out) + self.bias1)
        out = self.liner2(out) + self.bias2
        return out


if __name__ == '__main__':
    # vocab = Vocab(["sample_text.txt"])
    # vocab.cal_totals()
    # vocab.filter_min_freq()
    # vocab.write_vocab()


    tokenizer = Tokenizer("vocab_jayliu.txt")
    nnlm = NNLM(len(tokenizer.word_2_ix), 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nnlm.parameters(), lr=0.001)


    for epoch in range(100):
        dataset = Dataset(["sample_text.txt"], tokenizer)
        for i in dataset.get_batch():
            batch_X = torch.Tensor([k[0] for k in i]).int()
            batch_y = torch.Tensor([k[1] for k in i]).int()
            optimizer.zero_grad()
            out = nnlm(batch_X)
            print(out.shape, batch_y.shape)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        












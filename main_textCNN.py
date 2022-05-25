import torch
import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
import torch.nn as nn
import torch.optim as optim


label_map = {
        "100": 0,   # 民生 故事 news_story
        "101": 1,   # 文化 文化 news_culture
        "102": 2,   # 娱乐 娱乐 news_entertainment
        "103": 3,   # 体育 体育 news_sports
        "104": 4,   # 财经 财经 news_finance
        "106": 5,   # 房产 房产 news_house
        "107": 6,   # 汽车 汽车 news_car
        "108": 7,   # 教育 教育 news_edu
        "109": 8,   # 科技 科技 news_tech
        "110": 9,   # 军事 军事 news_military
        "112": 10,   # 旅游 旅游 news_travel
        "113": 11,   # 国际 国际 news_world
        "114": 12,   # 证券 股票 stock
        "115": 13,   # 农业 三农 news_agriculture
        "116": 14,   # 电竞 游戏 news_game
    }


class Corpus:
    def __init__(self, pth, batch_size=64):
        self.f = open(pth, encoding="utf-8")
        self.batch_size = batch_size

    def get_batch(self):
        flg = True
        batch = []
        while flg:
            try:
                i = next(self.f).strip().split("_!_")
                filtered = re.sub('\W*', '', i[3])
                slice_words = list(jieba.cut(filtered))
                vocab.read_sentence(slice_words)
                key_words = [k for k in i[4].split(",") if k]
                vocab.read_sentence(key_words)
                label = label_map[i[1]]
                batch.append([label, slice_words, key_words])
                if len(batch) >= 64:
                    yield batch
                    batch = []
            except StopIteration:
                flg = False
                yield batch

class Vocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.counter = Counter()
        self.unk_word = "<UNK>"
        self.pad_word = "<PAD>"

    def read_sentence(self, sen):
        for word in sen:
            self.counter[word] += 1

    def create_word_2_ix(self):
        word_2_ix = dict()
        rs = sorted(self.counter.items(), key=lambda x: x[1], reverse=True)
        word_2_ix[self.unk_word] = 0
        word_2_ix[self.pad_word] = 1
        shift = 2
        for ix, i in enumerate(rs):
            if i[1] <= self.min_freq:
                break
            word_2_ix[i[0]] = ix + shift
        self.word_2_ix = word_2_ix
        return word_2_ix


class Tokenizer:
    def __init__(self, vocab, max_len=20):
        self.vocab = vocab
        self.max_len = max_len

    def to_tokens(self, words):
        rs = []
        for w in words:
            if w not in self.vocab.word_2_ix:
                rs.append(self.vocab.word_2_ix[self.vocab.unk_word])
            else:
                rs.append(self.vocab.word_2_ix[w])
        if len(rs) > self.max_len:
            rs = rs[: self.max_len]
        else:
            for _ in range(self.max_len - len(rs)):
                rs.append(self.vocab.word_2_ix[self.vocab.pad_word])
        return rs


class TextCNN(nn.Module):
    def __init__(self, embed_size, embed_dim, num_class):
        super(TextCNN, self).__init__()
        self.emb = nn.Embedding(embed_size, embed_dim)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, embed_dim))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 1))
        self.flattern = nn.Flatten()
        self.linear = nn.Linear(240, num_class)
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = self.emb(x)
        x = torch.unsqueeze(x, -1)
        x = torch.permute(x, [0, 3, 1, 2])
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.flattern(out)
        out = self.linear(out)
        out = self.softmax(out)
        return out




if __name__ == '__main__':
    vocab = Vocab()
    corpus_pth = r"D:\my_ml_jayliu\data\toutiao_dataset\toutiao_cat_data.txt"

    # g = corpus.get_batch()
    # while True:
    #     try:
    #         rs = next(g)
    #         for i in rs:
    #             vocab.read_sentence(i[1])
    #         vocab.create_word_2_ix()
    #     except StopIteration:
    #         break
    # with open('vocab_toutiao.pickle', 'wb') as f:
    #     pickle.dump(vocab, f)

    with open ('vocab_toutiao.pickle', "rb") as f:
        vocab = pickle.load(f)


    EMBEDDING_SIZE = len(vocab.word_2_ix)
    EMBEDDING_DIM = 64
    NUM_CLASS =len(label_map)
    loss = nn.CrossEntropyLoss()
    textcnn = TextCNN(EMBEDDING_SIZE, EMBEDDING_DIM, NUM_CLASS)
    optimzer = optim.Adam(textcnn.parameters(), lr=0.0001)

    tokenize = Tokenizer(vocab)
    for epoch in range(20):
        corpus = Corpus(corpus_pth)
        g = corpus.get_batch()
        while True:
            try:
                batch = next(g)
                for ix, b in enumerate(batch):
                    tokens = tokenize.to_tokens(b[1])
                    b[1] = tokens
                    batch[ix] = b
                random.shuffle(batch)
                batch_x = torch.Tensor([i[1] for i in batch]).int()
                batch_y = torch.Tensor([i[0] for i in batch]).long()
                # print(batch_x)
                # print(batch_y)
                out = textcnn(batch_x)
                # print(batch_x.shape, out.shape)
                ls = loss(out, batch_y)
                optimzer.zero_grad()
                ls.backward()
                optimzer.step()
            except StopIteration:
                break
        print(epoch, ls)
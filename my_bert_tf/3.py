import tensorflow as tf
import collections
import random
import tokenization
tf.logging.set_verbosity(tf.logging.DEBUG)
tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)

class Instances:
    def __init__(self, file_pth, rng, max_length=128):
        self.file_pth = file_pth
        self.max_length = max_length - 3
        self.all_doc = []
        self.instances = []
        self.rng = rng

    def create_tokens_document(self):
        self.all_doc = []
        f = open(self.file_pth, encoding="utf-8")
        doc = []
        for i in f:
            i = i.strip()
            if not i:
                self.all_doc.append(doc)
                doc = []
                continue
            doc.append(tokenizer.tokenize(i))
        if doc:
            self.all_doc.append(doc)
        self.rng.shuffle(self.all_doc)
        f.close()

    def create_pair(self, doc_ix):
        doc = self.all_doc[doc_ix]
        ix = 0
        chunk = []
        chunk_length = 0
        while ix < len(doc):
            chunk.append(doc[ix])
            chunk_length += len(doc[ix])
            tf.logging.debug(f"DOC IX {ix}")
            if ix == len(doc) - 1 or chunk_length >= self.max_length:
                if chunk:
                    # split_ix = 1 if len(chunk) < 2 else self.rng.randint(1, len(chunk) - 1)
                    split_ix = 1  # 指的是这个文档如果只有两句话，终止到句子索引1，否则在第一句话之后的句子索引里面采样
                    if len(chunk) >= 2:
                        split_ix = rng.randint(1, len(chunk) - 1)
                    token_a = []
                    for ix in range(split_ix):
                        token_a.extend(chunk[ix])

                    token_b = []
                    is_next = True
                    if len(chunk) == 1 or self.rng.random() < 0.5:
                        is_next = False
                        max_token_b_length = self.max_length - len(token_a)
                        while True:
                            rand_doc_ix = self.rng.randint(0, len(self.all_doc) - 1)
                            if rand_doc_ix != doc_ix:
                                break

                        rand_doc = self.all_doc[rand_doc_ix]
                        rand_start = self.rng.randint(0, len(rand_doc) - 1)
                        for i in range(rand_start, len(rand_doc)):
                            token_b.extend(rand_doc[i])
                            if len(token_b) >= max_token_b_length:
                                break
                        # num_segment = len(chunk) - split_ix
                        # ix -= num_segment
                        # print(ix, num_segment, len(chunk), split_ix)
                    else:
                        for i in range(split_ix, len(chunk)):
                            token_b.extend(chunk[i])
                    tf.logging.debug(f"{token_a}")
                    tf.logging.debug(f"{token_b}")
                    tf.logging.debug(f"{is_next}***{ix}****{len(doc)}")


                chunk = []
                chunk_length = 0
            print(ix)
            ix += 1
            print(ix, len(doc))








if __name__ == '__main__':

    rng = random.Random()
    instances = Instances("sample_text.txt", rng)
    instances.create_tokens_document()
    print(instances.all_doc[0])
    instances.create_pair(0)






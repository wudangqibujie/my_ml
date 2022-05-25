import tensorflow as tf
import collections
import random
import tokenization

tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)

class Instances:
    def __init__(self, file_pth, rng):
        self.file_pth = file_pth
        self.instances = []
        self.rng = rng

    def create_tokens_document(self):
        all_doc = []
        f = open(self.file_pth, encoding="utf-8")
        doc = []
        for i in f:
            i = i.strip()
            if not i:
                all_doc.append(doc)
                doc = []
                continue
            doc.append(tokenizer.tokenize(i))
        if doc:
            all_doc.append(doc)
        self.rng.shuffle(all_doc)
        f.close()







rng = random.Random()
instances = Instances("sample_text.txt", rng)
instances.create_tokens_document()







if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)


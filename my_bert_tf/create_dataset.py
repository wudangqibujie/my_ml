import tensorflow as tf
import collections
import random
import tokenization

tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)

class Instances:
    def __init__(self, file_pth, rng, max_seq_len=128):
        self.file_pth = file_pth
        self.max_seq_len = max_seq_len - 3
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
        for i in self.all_doc:
            tf.logging.info(f"{i}")
        self.rng.shuffle(self.all_doc)
        f.close()

    def create_instance(self):
        for ix in self.all_doc:
            samples = self.create_samples_from_doc(ix)
            self.instances.extend(samples)
        self.rng.shuffle(self.instances)

    def truncate_sent(self, token_a, token_b):
        pass

    def create_pair_samples_from_doc(self, doc_ix):
        doc = self.all_doc[doc_ix]
        rslt = []
        ix = 0
        chunk = []
        chunk_len = 0
        while ix < len(doc):
            chunk_len += len(doc[ix])
            chunk.append(doc[ix])
            tf.logging.debug(f"current doc_idx {ix} chunk len {chunk_len}")
            if ix == len(doc) - 1 or chunk_len >= self.max_seq_len:
                if chunk:
                    # 先随机选取一个切分点
                    split_point = 1 if len(chunk) <= 2 else self.rng.randint(1, len(chunk) - 1)

                    sent_a = []
                    for i in range(split_point):
                        sent_a.extend(chunk[i])

                    sent_b = []
                    # 需要进行是否下一句样本构造
                    is_next = True
                    if self.rng.random() > 0.5 or len(chunk) == 1:
                        is_next = False
                        b_len_needed = self.max_seq_len - len(sent_a)
                        # 随机选择一个doc
                        while True:
                            rnd_doc_ix = self.rng.randint(0, len(self.all_doc) - 1)
                            if rnd_doc_ix != doc_ix:
                                break
                        rnd_doc = self.all_doc[rnd_doc_ix]
                        rand_start = self.rng.randint(0, len(rnd_doc) - 1)
                        for i in range(rand_start, len(rnd_doc)):
                            sent_b.extend(rnd_doc[i])
                            if len(sent_a) >= b_len_needed:
                                break

                        ix = ix - len(chunk) + split_point
                    else:
                        for i in range(split_point, len(chunk)):
                            sent_b.extend(doc[i])
                    tf.logging.debug(f"{sent_a}")
                    tf.logging.debug(f"{sent_b}")
                    tf.logging.debug(f"{len(sent_a)} - {len(sent_b)}, is_next:{is_next}")
                    # self.truncate_sent(sent_a, sent_b)
                    tokens = []
                    segment_ids = []
                    tokens.append("[CLS]")
                    segment_ids.append(0)
                    for i in sent_a:
                        tokens.append(i)
                    for _ in range(len(sent_a)):
                        segment_ids.append(0)
                    tokens.append("[SEP]")
                    segment_ids.append(0)
                    for i in sent_b:
                        tokens.append(i)
                    for _ in range(len(sent_b)):
                        segment_ids.append(1)
                    tokens.append("[SEP]")
                    segment_ids.append(1)
                    rslt.append((tokens, segment_ids, is_next))
                chunk_len = 0
                chunk = []
            ix += 1
        return rslt

    def mask_lm_sample(self, tokens):
        mask_idxes = []
        mask_labels = []
        return mask_idxes, mask_labels

    def create_samples_from_doc(self, doc_ix):
        tf.logging.debug(f"DOC :{doc_ix}, has {len(self.all_doc[doc_ix])} sentences")
        pairs_samples = self.create_pair_samples_from_doc(doc_ix)
        for tokens, segment_ids, is_next in pairs_samples:
            mask_idxes, mask_labels = self.mask_lm_sample(tokens)
            self.instances.append({
                "tokens": tokens,
                "segment_ids": segment_ids,
                "is_next": is_next,
                "mask_lm_idx": mask_idxes,
                "mask_lm_labels": mask_labels
            })



tf.logging.set_verbosity(tf.logging.DEBUG)
rng = random.Random()
instances = Instances("sample_text.txt", rng)
instances.create_tokens_document()
print(instances.all_doc)
instances.create_samples_from_doc(0)


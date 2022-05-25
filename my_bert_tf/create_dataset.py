import tensorflow as tf
import collections
import random
import tokenization

tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)

class Instances:
    def __init__(self, file_pth, rng, max_seq_len=128, max_mask_per_token=20, mask_rt=0.15):
        self.file_pth = file_pth
        self.max_mask_per_tokens = max_mask_per_token
        self.mask_rt = mask_rt
        self.max_seq_len = max_seq_len - 3
        self.all_doc = []
        self.instances = []
        self.rng = rng
        self.vocab = tokenizer.vocab
        self.vocab_lst = list(tokenizer.vocab.keys())

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
        now_length = len(token_a) + len(token_b)
        if now_length <= self.max_seq_len:
            return
        longer_token = token_a if len(token_a) > len(token_b) else token_b
        short_len = now_length - len(longer_token)
        font_or_not = self.rng.random() < 0.5
        while short_len + len(longer_token) > self.max_seq_len:
            if font_or_not:
                del longer_token[0]
            else:
                longer_token.pop()


    def create_pair_samples_from_doc(self, doc_ix):
        doc = instances.all_doc[doc_ix]
        tf.logging.debug(f"{doc}")
        pairs_info = []
        chunk = []
        chunk_len = 0
        ix = 0
        while ix < len(doc):
            seg = doc[ix]
            chunk.append(seg)
            chunk_len += len(seg)
            if ix == len(doc) - 1 or chunk_len >= self.max_seq_len:
                tf.logging.debug(3 * f"* now seg_ix {ix} **************************************************")
                tf.logging.debug(f"chunk: **{chunk_len}**{[len(i) for i in chunk]}**{chunk}**")
                cut_idx = rng.randint(1, len(chunk) - 1) if len(chunk) >= 2 else 1
                tf.logging.debug(f"cut_ix: ****{cut_idx}***")
                token_a = []
                for i in range(cut_idx):
                    token_a.extend(chunk[i])
                tf.logging.debug(f"token_a: *{len(token_a)}****{token_a}***")
                token_b = []

                is_next = rng.random() < 0.5
                tf.logging.debug(f"is_next: ******{is_next}******")
                if len(chunk) == 1 or not is_next:
                    needed_length = self.max_seq_len - len(token_a)
                    tf.logging.debug(f"is_not_next, sample_token_b needed_len :  {needed_length}")
                    while True:
                        rnd_doc_ix = rng.randint(0, len(instances.all_doc) - 1)
                        if rnd_doc_ix != doc_ix:
                            break
                    tf.logging.debug(f"sample_negative_doc_ix:  {rnd_doc_ix}")
                    rnd_doc = instances.all_doc[rnd_doc_ix]
                    tf.logging.debug(f"sample_negative_doc:  {[len(i) for i in rnd_doc]}")
                    rnd_start_ix = rng.randint(0, len(rnd_doc) - 1)
                    tf.logging.debug(f"rnd doc-rnd_start_ix:  {rnd_start_ix}")
                    for seg_ix in range(rnd_start_ix, len(rnd_doc)):
                        token_b.extend(rnd_doc[seg_ix])
                        if len(token_b) >= needed_length:
                            break
                    tf.logging.debug(f"sampled_token_b :  **{len(token_b)}***{token_b}")
                    ix = ix - (len(chunk) - cut_idx)
                    tf.logging.debug(f"reset ix : {ix} ")
                else:
                    for i in range(cut_idx, len(chunk)):
                        token_b.extend(chunk[i])
                tf.logging.debug(f"token_b: *{len(token_b)}****{token_b}***")

                tf.logging.debug(f"token total len:  {len(token_a) + len(token_b)}***{len(token_a)}**{len(token_b)}**")
                tf.logging.debug(f"*************  Truncate tokens  **************")
                instances.truncate_sent(token_a, token_b)
                tf.logging.debug(
                    f"truncated total token len :**** {len(token_a) + len(token_b)}***{len(token_a)}**{len(token_b)}**")

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)

                for i in token_a:
                    tokens.append(i)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in token_b:
                    tokens.append(i)
                    segment_ids.append(1)

                tokens.append("[SEP]")
                segment_ids.append(1)

                tf.logging.debug(f"tokens len:  {len(tokens)}  **{tokens}**")
                tf.logging.debug(f"segment len:  {len(segment_ids)}  **{segment_ids}**")

                pairs_info.append((tokens, segment_ids, int(is_next)))

                chunk = []
                chunk_len = 0
            ix += 1
        return pairs_info

    def mask_lm_sample(self, tokens):
        """
        一个token有15%被mask， 设置最大mask数目
        mask的其中
            80% mask
            10% 其他token
            10% 原来token
        :param tokens:
        :return:
        """
        tf.logging.debug(5 * f"************ MASK **************")
        mask_num = min(self.max_mask_per_tokens, int(len(tokens) * self.mask_rt))
        tf.logging.debug(f"token len {len(tokens)}**** mask num {mask_num} ")
        mask_idxes = []
        mask_tokens = []
        mask_label_id = []
        word_idxes = []
        for ix, i in enumerate(tokens):
            if i in ["[CLS]", "[SEP]"]:
                continue
            if i.startswith("##"):
                word_idxes[-1].append(ix)
            word_idxes.append([ix])
        tf.logging.debug(f"**{tokens}**")
        tf.logging.debug(f"word_idxes:  *** {word_idxes} ***")

        mask_num = min(self.max_mask_per_tokens, max(1, int(len(word_idxes) * self.mask_rt)))
        tf.logging.debug(f"mask num {mask_num}")

        self.rng.shuffle(word_idxes)

        stop_flg = 0
        masked_index = set()
        while stop_flg < mask_num:
            rnd_word_ixes = word_idxes[self.rng.randint(0, len(word_idxes) - 1)]
            if str(rnd_word_ixes) in masked_index:
                continue
            rnd_mask_type = self.rng.random()
            if rnd_mask_type < 0.8:
                # mask
                for i in rnd_word_ixes:
                    mask_tokens.append(tokens[i])
                    mask_label_id.append(self.vocab[tokens[i]])
                    tokens[i] = "[MASK]"
                    mask_idxes.append(i)
            else:
                if self.rng.random() < 0.5:
                    # do not mask
                    for i in rnd_word_ixes:
                        # tokens[i] = tokens[i]
                        mask_label_id.append(self.vocab[tokens[i]])
                        mask_idxes.append(i)
                        mask_tokens.append(tokens[i])
                else:
                    # oth word
                    for i in rnd_word_ixes:
                        mask_tokens.append(tokens[i])
                        mask_label_id.append(self.vocab[tokens[i]])
                        rs = self.vocab_lst[self.rng.randint(0, len(self.vocab_lst) - 1)]
                        while rs in ["[CLS]", "[SEP]"]:
                            rs = self.vocab_lst[self.rng.randint(0, len(self.vocab_lst) - 1)]
                        tokens[i] = rs
                        mask_idxes.append(i)
            stop_flg += 1
            masked_index.add(str(rnd_word_ixes))
        masked_lm_weights = [1.0 for _ in range(len(mask_label_id))]
        if len(mask_label_id) < self.max_mask_per_tokens:
            for _ in range(self.max_mask_per_tokens - len(mask_label_id)):
                masked_lm_weights.append(0.)
                mask_label_id.append(0)
                mask_idxes.append(0)

        tf.logging.debug(f"Mask tokens: {len(mask_tokens)} ***{mask_tokens}**")
        tf.logging.debug(f"Mask index:  {len(mask_idxes)} ***{mask_idxes}**")
        tf.logging.debug(f"Mask label id:  {len(mask_label_id)} ***{mask_label_id}**")

        tf.logging.debug(f"Masked labels: ***{tokens}**")
        return mask_idxes, mask_tokens, mask_label_id, masked_lm_weights

    def create_samples_from_doc(self, doc_ix):
        tf.logging.debug(f"DOC :{doc_ix}, has {len(self.all_doc[doc_ix])} sentences")
        pairs_samples = self.create_pair_samples_from_doc(doc_ix)
        for tokens, segment_ids, is_next in pairs_samples:
            mask_idxes, mask_tokens, mask_label_id, masked_lm_weights = self.mask_lm_sample(tokens)
            input_id = [1 for _ in range(len(tokens))]
            token_id = tokenizer.convert_tokens_to_ids(tokens)
            if len(input_id) < self.max_seq_len:
                for _ in range(self.max_seq_len - len(input_id)):
                    input_id.append(0)
                    token_id.append(0)
            tf.logging.debug(f"token_id: ** {len(token_id)} **  {token_id}")
            tf.logging.debug(f"input_id: ** {len(input_id)} **  {input_id}")

            self.instances.append({
                "tokens": tokens,
                "token_id": tokens,
                "segment_ids": segment_ids,
                "is_next": is_next,
                "mask_lm_idx": mask_idxes,
                "mask_lm_labels": mask_label_id,
                "mask_lm_weights": masked_lm_weights,
                "mask_tokens": mask_tokens,
                "input_id": input_id
            })

        for i in self.instances:
            tf.logging.info(6 * f"*****sample******")
            for k, v in i.items():
                tf.logging.info(f"{k}--{v}")

class CorpusTF:
    def __init__(self, pths):
        self.writers = [tf.python_io.TFRecordWriter(i) for i in pths]

    def _to_int_feature(self, data):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(data)))

    def _to_float_feature(self, data):
        return f.train.Feature(float_list=tf.train.FloatList(value=list(data)))

    def write_insatnce(self, sample):
        features = collections.OrderedDict()
        features["token_id"] = self._to_int_feature(sample["token_id"])
        features["segment_ids"] = self._to_int_feature(sample["segment_ids"])
        features["is_next"] = self._to_int_feature(sample["is_next"])
        features["mask_lm_idx"] = self._to_int_feature(sample["mask_lm_idx"])
        features["mask_lm_labels"] = self._to_int_feature(sample["mask_lm_labels"])
        features["input_id"] = self._to_int_feature(sample["input_id"])
        features["input_id"] = self._to_float_feature(sample["mask_lm_weights"])
        tf_sample = tf.train.Example(features=tf.train.Features(feature=features))



tf.logging.set_verbosity(tf.logging.DEBUG)
rng = random.Random(100)
instances = Instances("sample_text.txt", rng)
instances.create_tokens_document()
print(instances.all_doc)
# instances.all_doc.append([["as", "asd", "dwd"]])
for doc in instances.all_doc:
    print(len(doc), sum([len(i) for i in doc]), [len(i) for i in doc], doc)
doc_ix = 0
instances.create_samples_from_doc(doc_ix)




# split_ix = rng.randint(1, len(doc) - 1)
# tf.logging.info(f"split_idx:{split_ix}")



# INFO:tensorflow:*** Example ***
# INFO:tensorflow:tokens: [CLS] possibly this may have been the reason why early rise ##rs in that locality , during the rainy season , [MASK] a thoughtful habit of body , [MASK] seldom [MASK] their eyes to the rift ##ed or india - ink washed skies above them . " cass [MASK] [MASK] had risen early that morning , but not with a [MASK] to discovery . [SEP] [MASK] [MASK] frederick ##l of the [MASK] , the [MASK] stream of [MASK] faces , the line of cu ##rri [MASK] , pal ##an kneeling ##s , laden [MASK] ##es , camel ##s [MASK] elephants , which met and passed him , and [MASK] him up steps and [MASK] doorway ##s , [MASK] they threaded their way through the great moon - [SEP]
# INFO:tensorflow:input_ids: 101 4298 2023 2089 2031 2042 1996 3114 2339 2220 4125 2869 1999 2008 10246 1010 2076 1996 16373 2161 1010 103 1037 16465 10427 1997 2303 1010 103 15839 103 2037 2159 2000 1996 16931 2098 2030 2634 1011 10710 8871 15717 2682 2068 1012 1000 16220 103 103 2018 13763 2220 2008 2851 1010 2021 2025 2007 1037 103 2000 5456 1012 102 103 103 5406 2140 1997 1996 103 1010 1996 103 5460 1997 103 5344 1010 1996 2240 1997 12731 18752 103 1010 14412 2319 16916 2015 1010 14887 103 2229 1010 19130 2015 103 16825 1010 2029 2777 1998 2979 2032 1010 1998 103 2032 2039 4084 1998 103 7086 2015 1010 103 2027 26583 2037 2126 2083 1996 2307 4231 1011 102
# INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# INFO:tensorflow:masked_lm_positions: 21 28 30 48 49 60 65 66 67 71 74 77 85 89 93 98 108 113 117 0
# INFO:tensorflow:masked_lm_ids: 4233 1998 4196 1000 10154 3193 1998 1059 11961 2395 18870 5697 18954 12519 4632 1010 7757 2046 2004 0
# INFO:tensorflow:masked_lm_weights: 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0
# INFO:tensorflow:next_sentence_labels: 1

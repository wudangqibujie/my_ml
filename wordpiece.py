

vocab = ["att", "##ir", "##bute", "##tirbute", "a", "##attirbute"]

def wordpiece(token):
    # whitespace_tokenize是先将text按照空格切分，这对于输入一个句子的情况下有用
    # 接下来，把token想象成单词“unaffable”
    chars = list(token)
    is_bad = False
    start = 0
    sub_tokens = []
    # 初始化了start
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            # 从最后一个字符逐个向左遍历，保证匹配到的子词是最长的
            substr = "".join(chars[start:end])
            print(substr)
            if start > 0:  # 添加特殊的连接符
                substr = "##" + substr
            if substr in vocab:  # 姑且把vocab理解为一个列表或键为词表中单词的字典
                cur_substr = substr
                break

            end -= 1
        print(cur_substr, "*********")
        if cur_substr is None:
            # 这里是一个否决条件，如果end走了一遍仍没有找到合适的子词，那么说明当前从start到end组成的子词不在词表中
            is_bad = True
            break
        sub_tokens.append(cur_substr)
        start = end
    print(sub_tokens)


wordpiece("attirbute")
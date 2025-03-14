# -*- coding: utf-8 -*-
import json

import numpy as np
from transformers import BertTokenizerFast


def extract_spoes(config_paras, id2schema, text, tokenizer, logits1, logits2, logits3):
    logits1 = logits1.cpu()
    logits2 = logits2.cpu()
    logits3 = logits3.cpu()
    fw = open("./predict.jsonl", "a", encoding="utf-8")
    token2char_span_mapping = \
    tokenizer(text, return_offsets_mapping=True, max_length=int(config_paras['maxlen']), truncation=True)[
        "offset_mapping"]
    threshold = 0.0
    subjects, objects = set(), set()
    logits1[:, [0, -1]] -= np.inf
    logits1[:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(logits1 > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(logits2[:, sh, oh] > threshold)[0]
            p2s = np.where(logits3[:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                spoes.add((
                    text[token2char_span_mapping[sh][0]:token2char_span_mapping[st][-1]], id2schema[p].split("_")[1],
                    text[token2char_span_mapping[oh][0]:token2char_span_mapping[ot][-1]]
                ))
    s = json.dumps({'text': text, 'spo_list': list(spoes)}, ensure_ascii=False)
    fw.write(s + "\n")
    return list(spoes)


class Evaluate:
    def __init__(self):
        pass

    @staticmethod
    def evaluate_train(config_paras, id2schema, tokenizer, batch_text, train_text, logits1, logits2, logits3):
        """
        评估函数，计算f1、precision、recall
        """
        num = 0
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for text in batch_text:
            # 句子分词并生成位置信息
            spoes = extract_spoes(config_paras, id2schema, text, tokenizer, logits1[num], logits2[num], logits3[num])
            num = num + 1
            text_num = 0
            for index, value in enumerate(train_text):
                if value['text'] == text:
                    text_num = index
            R = set([SPO(spo) for spo in spoes])
            T = set([SPO(spo) for spo in train_text[text_num]['spo_list']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)

        return X, Y, Z


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        tokenizer = BertTokenizerFast.from_pretrained('./roberta-large', do_lower_case=True)
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


if __name__ == '__main__':
    handler = SPO()
    print(handler.__hash__())

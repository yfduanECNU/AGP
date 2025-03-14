# -*- coding: utf-8 -*-

import sys
import numpy as np
import torch
import configparser

from transformers import BertTokenizerFast


sys.path.insert(0, '')
sys.path.insert(0, '..')
from model.bert_optimization import BertAdam


class SetOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def set_optimizer(model, train_steps=None):
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=2e-5,
                             warmup=0.1,
                             t_total=train_steps)
        return optimizer


class Loss_function:
    def __init__(self):
        pass

    @staticmethod
    def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
        '''
        稀疏多标签交叉熵损失的torch实现
        '''
        shape = y_pred.shape
        y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
        y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if mask_zero:
            infs = zeros + 1e12
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
        aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
        neg_loss = all_loss + torch.log(aux_loss)
        loss = torch.mean(torch.sum(pos_loss + neg_loss))
        return loss


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        con = configparser.ConfigParser()
        con.read('./config.ini', encoding='utf8')
        args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
        tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)
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
    handle_Load_config = SetOptimizer()

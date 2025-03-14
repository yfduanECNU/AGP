# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from utils.data_generator import Data_generator
from model.erenet import ERENet
from model.globalpointer import RawGlobalPointer
from utils.dataprocess import DataProcessor
from utils.evaluate import Evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Predict:
    def __init__(self):
        pass

    @staticmethod
    def test(mode, n_fold, config_paras, tokenizer, encoder, schema, id2schema):
        mention_detect = RawGlobalPointer(hiddensize=int(config_paras['plm_hidden_size']),
                                          ent_type_size=int(config_paras['ent_type_size']),
                                          inner_dim=int(config_paras['inner_dim'])).to(device)
        s_o_head = RawGlobalPointer(hiddensize=int(config_paras['plm_hidden_size']),
                                    ent_type_size=len(schema),
                                    inner_dim=int(config_paras['inner_dim']),
                                    RoPE=False, tril_mask=False).to(device)
        s_o_tail = RawGlobalPointer(hiddensize=int(config_paras['plm_hidden_size']),
                                    ent_type_size=len(schema),
                                    inner_dim=int(config_paras['inner_dim']),
                                    RoPE=False, tril_mask=False).to(device)
        net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)

        if mode == 'dev':
            data_file = '{}{}/{}/dev.jsonl'.format(config_paras['dataset_path'], config_paras['dataset'], n_fold)
            net.load_state_dict(torch.load('erenet.pth'))
        else:
            data_file = '{}{}/{}/test.jsonl'.format(config_paras['dataset_path'], config_paras['dataset'], n_fold)
            net.load_state_dict(torch.load('best1.pth'))

        valid_data = DataProcessor.get_filedata(data_file)
        input_data = Data_generator(DataProcessor.get_filedata(data_file),
                                   tokenizer, max_len=int(config_paras["maxlen"]), schema=schema)
        data_loader = DataLoader(input_data, batch_size=int(config_paras["batch_size"]), shuffle=True,
                                 collate_fn=input_data.collate)

        net.eval()
        total_X, total_Y, total_Z = 0., 0., 0.
        for idx, batch in enumerate(data_loader):
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(
                    device), batch_entity_labels.to(
                    device), batch_head_labels.to(device), batch_tail_labels.to(device)
            logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            # 计算F1
            X, Y, Z = Evaluate.evaluate_train(config_paras, id2schema, tokenizer, text, valid_data, logits1, logits2,
                                              logits3)
            total_X += X
            total_Y += Y
            total_Z += Z

        f1, precision, recall = 2 * total_X / (total_Y + total_Z), total_X / total_Y, total_X / total_Z
        # print('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))

        return f1, precision, recall

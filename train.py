# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel

from predict import Predict
from utils.data_generator import Data_generator
from model.erenet import ERENet
from model.globalpointer import RawGlobalPointer
from model.attack_train import PGD
from utils.dataprocess import DataProcessor
from utils.common import SetOptimizer, Loss_function


class Trainer:
    def __init__(self, n_fold):
        self.n_fold = n_fold

    def train(self):
        # 设置运行设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(1)

        # 加载config和schema
        config_path = 'config.ini'
        config_paras = DataProcessor.get_configparas(config_path)
        schema_path = '{}{}/schemas.json'.format(config_paras['dataset_path'], config_paras['dataset'])
        schema, id2schema = DataProcessor.schema2id(schema_path)
        # 设置分词器
        tokenizer = BertTokenizerFast.from_pretrained(config_paras['model_path'], do_lower_case=True)
        encoder = BertModel.from_pretrained(config_paras['model_path'])

        # n_fold轮训练和测试
        train_file_path = '{}{}/{}/train.jsonl'.format(config_paras['dataset_path'], config_paras['dataset'], self.n_fold)

        # 按batch size大小加载数据
        train_data = Data_generator(DataProcessor.get_filedata(train_file_path), tokenizer,
                                   max_len=int(config_paras['maxlen']), schema=schema)
        train_loader = DataLoader(train_data, batch_size=int(config_paras['batch_size']), shuffle=True,
                                  collate_fn=train_data.collate)

        # 通过GlobalPointer进行采样
        mention_detect = RawGlobalPointer(hiddensize=int(config_paras['plm_hidden_size']),
                                          ent_type_size=int(config_paras['ent_type_size']),
                                          inner_dim=int(config_paras['inner_dim'])).to(device)  # 实体关系抽取任务默认不提取实体类型
        s_o_head = RawGlobalPointer(hiddensize=int(config_paras['plm_hidden_size']),
                                    ent_type_size=len(schema),
                                    inner_dim=int(config_paras['inner_dim']),
                                    RoPE=False, tril_mask=False).to(device)
        s_o_tail = RawGlobalPointer(hiddensize=int(config_paras['plm_hidden_size']),
                                    ent_type_size=len(schema),
                                    inner_dim=int(config_paras['inner_dim']),
                                    RoPE=False, tril_mask=False).to(device)
        net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
        net.train()
        pgd = PGD(model=net)
        # print(net)

        # 设置自适应优化器
        optimizer = SetOptimizer.set_optimizer(net,
                                               train_steps=(int(len(train_data) / int(
                                                   config_paras['batch_size'])) + 1) * int(
                                                   config_paras['epochs']))

        # 训练模型并进行保存
        best_f1 = float('-inf')
        early_stopping = 0
        train_loss_list = []
        dev_f1_list = []
        for eo in range(int(config_paras['epochs'])):
            epoch_iterator = tqdm(train_loader, desc='Training (Epoch %d)' % (eo + 1))
            epoch_loss = 0.0
            for idx, batch in enumerate(epoch_iterator):
                text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
                batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(
                        device), batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(
                        device)
                logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
                # 计算loss
                loss1 = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels,
                                                                                 y_pred=logits1,
                                                                                 mask_zero=True)
                loss2 = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels,
                                                                                 y_pred=logits2,
                                                                                 mask_zero=True)
                loss3 = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels,
                                                                                 y_pred=logits3,
                                                                                 mask_zero=True)
                loss = sum([loss1, loss2, loss3]) / 3
                loss.backward()

                # 对抗学习
                pgd_k = 3
                pgd.backup_grad()
                for _t in range(pgd_k):
                    pgd.attack(is_first_attack=(_t == 0))

                    if _t != pgd_k - 1:
                        net.zero_grad()
                    else:
                        pgd.restore_grad()

                    logits1_adv, logits2_adv, logits3_adv = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
                    loss1_adv = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels,
                                                                                         y_pred=logits1_adv,
                                                                                         mask_zero=True)
                    loss2_adv = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels,
                                                                                         y_pred=logits2_adv,
                                                                                         mask_zero=True)
                    loss3_adv = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels,
                                                                                         y_pred=logits3_adv,
                                                                                         mask_zero=True)
                    loss_adv = sum([loss1_adv, loss2_adv, loss3_adv]) / 3
                    loss_adv.backward()
                pgd.restore()

                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

            train_loss_list.append(epoch_loss)
            torch.save(net.state_dict(), 'erenet.pth')

            # dev
            # print("dev")
            dev_f1, dev_precision, dev_recall = Predict.test('dev', self.n_fold, config_paras, tokenizer, encoder, schema,
                                                             id2schema)
            dev_f1_list.append(dev_f1)
            print(
                'epoch:{}, loss:{}, f1:{}, precision:{}, recall:{}'.format((eo + 1), epoch_loss, dev_f1, dev_precision,
                                                                           dev_recall))

            if dev_f1 > best_f1:
                torch.save(net.state_dict(), 'best.pth')
                best_f1 = dev_f1
                early_stopping = 0
            else:
                early_stopping += 1

        # test
        # print("test")
        test_f1, test_precision, test_recall = Predict.test('test', self.n_fold, config_paras, tokenizer, encoder,
                                                            schema, id2schema)

        print('train_loss={}'.format(train_loss_list))
        print('dev_F1={}\nmax(dev_F1)={}'.format(dev_f1_list, max(dev_f1_list)))
        print('test_F1:{}, test_precision:{}, test_recall:{}\n'.format(test_f1, test_precision, test_recall))

        return train_loss_list, dev_f1_list, test_f1

#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
main
======
The main class for relation extration.
"""
import datetime
import sys
from train import Trainer
from utils.common import Logger
from utils.dataprocess import DataProcessor


def main_proc(n_folds):
    train_losses_list, dev_f1_scores_list, test_f1_scores_list = [], [], []
    for n in range(n_folds):
        print("This is the {} cross fold.".format(n + 1))

        train_losses, dev_f1_scores, test_f1_scores = Trainer(n).train()

        train_losses_list.append(train_losses)
        dev_f1_scores_list.append(dev_f1_scores)
        test_f1_scores_list.append(test_f1_scores)

    print("\ntrain_losses: {}\n, dev_f1: {}\n, test_f1: {}\n".format(train_losses_list, dev_f1_scores_list,
                                                                       test_f1_scores_list))


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    config_path = 'config.ini'
    config_paras = DataProcessor.get_configparas(config_path)
    sys.stdout = Logger('./log/{}_{}_{}_{}_pgd.log'.format(config_paras['dataset'], 'gplinker', config_paras['maxlen'],
                                                       config_paras['batch_size']),
                        sys.stdout)

    main_proc(n_folds=1)

    end_time = datetime.datetime.now()
    print('Takes {} seconds.'.format((end_time - start_time).seconds))
    print('Done main!')


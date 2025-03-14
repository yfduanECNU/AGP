# -*- coding: utf-8 -*-
"""
@Auth:
"""
import json
import configparser


class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def get_configparas(path):
        con = configparser.ConfigParser()
        con.read(path, encoding='utf8')
        config_paras = dict(dict(con.items('PLM')), **dict(con.items("para")), **dict(con.items("paths")))

        return config_paras

    @staticmethod
    def schema2id(path):
        schema2idx = {}
        with open(path, 'r', encoding='utf-8') as f:
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                schema2idx[item["subject_type"] + "_" + item["predicate"] + "_" + item["object_type"]] = idx

        idx2schema = {}
        for k, v in schema2idx.items():
            idx2schema[v] = k

        return schema2idx, idx2schema

    @staticmethod
    def get_filedata(filename):
        """加载数据
        单条格式：{'text': text, 'spo_list': [(s, p, o, s_t, o_t)]}
        """
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                data.append({
                    "text": line["text"],
                    "spo_list": [(spo["subject"], spo["predicate"], spo["object"]["@value"], spo["subject_type"],
                                  spo["object_type"]["@value"])
                                 for spo in line["spo_list"]]
                })
            return data

    @staticmethod
    def get_filedata_valid(filename):
        """加载数据
        单条格式：{'text': text, 'spo_list': [(s, p, o)]}
        """
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                data.append({
                    'text': line['text'],
                    'spo_list': [(spo['subject'], spo['predicate'], spo['object']["@value"])
                                 for spo in line['spo_list']]
                })
        return data



if __name__ == '__main__':
    handler = DataProcessor()
    re = handler.get_filedata('../datasets/CMeIE-V2_dev.jsonl')
    print(re[0])

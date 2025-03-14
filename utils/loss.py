import numpy as np
import torch


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
        zeros = torch.zeros_like(y_pred[...,:1])
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

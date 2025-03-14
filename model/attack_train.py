# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import clamp
from model.globalpointer import RawGlobalPointer
from utils.loss import Loss_function


# FGSM
class FGSM:
    def __init__(self, model: nn.Module, eps=0.1):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings)'):
        for name, param in self.model.named_parameters():

            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                r_at = self.eps * param.grad.sign()
                param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings)'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}


# FGM
class FGM:
    def __init__(self, model: nn.Module, eps=0.2):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}


# PGD
class PGD:
    def __init__(self, model, eps=0.1, alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


# FreeAT
class FreeAT:
    def __init__(self, model):
        self.model = model

    def attack(self, epsilon=1., delta=None):
        grad = delta.grad.detach()
        norm = torch.norm(grad)
        if norm != 0 and not torch.isnan(norm):
            delta.data = clamp(delta + epsilon * grad / norm, torch.tensor((-epsilon)).cuda(),
                               torch.tensor((epsilon)).cuda())
        return delta
    # def attack(self, epsilon=1., emb_name='embedding'):
    #     # emb_name这个参数要换成你模型中embedding的参数名
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and emb_name in name:
    #             norm = torch.norm(param.grad)
    #             if norm != 0 and not torch.isnan(norm):
    #                 r_at = clamp(epsilon * param.grad / norm, torch.tensor((-epsilon)).cuda(),
    #                              torch.tensor((epsilon)).cuda())
    #                 param.data.add_(r_at)
    # def __init__(self, model, eps=0.1):
    #     self.model = (
    #         model.module if hasattr(model, "module") else model
    #     )
    #     self.eps = eps
    #     self.emb_backup = {}
    #     self.grad_backup = {}
    #     self.last_r_at = 0
    #
    # def attack(self, emb_name='word_embeddings', is_first_attack=False):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and emb_name in name:
    #             if is_first_attack:
    #                 self.emb_backup[name] = param.data.clone()
    #             param.data.add_(self.last_r_at)
    #             param.data = self.project(name, param.data)
    #             self.last_r_at = self.last_r_at + self.eps * param.grad.sign()
    #
    # def restore(self, emb_name='word_embeddings'):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and emb_name in name:
    #             assert name in self.emb_backup
    #             param.data = self.emb_backup[name]
    #     self.emb_backup = {}
    #
    # def project(self, param_name, param_data):
    #     r = param_data - self.emb_backup[param_name]
    #     if torch.norm(r) > self.eps:
    #         r = self.eps * r / torch.norm(r)
    #     return self.emb_backup[param_name] + r
    #
    # def backup_grad(self):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and param.grad is not None:
    #             self.grad_backup[name] = param.grad.clone()
    #
    # def restore_grad(self):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and param.grad is not None:
    #             param.grad = self.grad_backup[name]



class FreeLB():
    def __init__(self, adv_K, adv_lr, adv_init_mag, adv_max_norm=0., adv_norm_type='l2'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag  # adv-training initialize with what magnitude, 即我们用多大的数值初始化delta
        self.adv_norm_type = adv_norm_type


    def attack(self, model, inputs,batch_entity_labels,batch_head_labels,batch_tail_labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = inputs['input_ids']
        #获取初始化时的embedding
        # embeds_init = getattr(model, self.encoder).embeddings.word_embeddings(input_ids.to(device))
        embeds_init = model.encoder.embeddings.word_embeddings(input_ids.to(device))

        if self.adv_init_mag > 0:   # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
        else:
            delta = torch.zeros_like(embeds_init)  # 扰动初始化
        # loss, logits = None, None


        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init  # 累积一次扰动delta
            # inputs['input_ids'] = None
            outputs= model.encoder(input_ids=None,
                            attention_mask=inputs["attention_mask"].to(device),
                             token_type_ids=inputs["token_type_ids"].to(device),
                            inputs_embeds=inputs["inputs_embeds"].to(device))

            s_o_head = RawGlobalPointer(hiddensize=768, ent_type_size=53, inner_dim=64, RoPE=False,
                                        tril_mask=False).to(
                device)
            mention_detect=RawGlobalPointer(hiddensize=768, ent_type_size=2, inner_dim=64, RoPE=False, tril_mask=False).to(
                device)

            logits1 = mention_detect(outputs, inputs["attention_mask"].to(device))
            logits2 = s_o_head(outputs, inputs["attention_mask"].to(device))
            logits3 = s_o_head(outputs, inputs["attention_mask"].to(device))
            loss1 = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1,
                                                                             mask_zero=True)
            loss2 = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2,
                                                                             mask_zero=True)
            loss3 = Loss_function.sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3,
                                                                             mask_zero=True)
            total_loss = sum([loss1, loss2, loss3]) / 3
            # print(loss)
            loss = total_loss / self.adv_K # 求平均的梯度

            loss.backward(retain_graph=True)

            if astep == self.adv_K - 1:
                # further updates on delta
                break

            delta_grad = delta.grad.clone().detach()  # 备份扰动的grad
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))

            embeds_init = model.encoder.embeddings.word_embeddings(input_ids.to(device))
        return loss


# SMART
# https://blog.csdn.net/HUSTHY/article/details/119005324
class SmartPerturbation():
    """
    step_size noise扰动学习率
    epsilon 梯度scale时防止分母为0
    norm_p 梯度scale采用的范式
    noise_var 扰动初始化系数
    loss_map 字典，loss函数的类型{"0":mse(),....}
    使用方法
    optimizer =
    model =
    loss_func =
    loss_map = {"0":loss_fun0,"1":loss_fun1,...}
    smart_adv = SmartPerturbation(model,epsilon,step_size,noise_var,loss_map)
    for batch_input, batch_label in data:
        inputs = {'input_ids':...,...,'labels':batch_label}
        logits = model(**inputs)
        loss = loss_func(logits,batch_label)
        loss_adv = smart_adv.forward(logits,input_ids,token_type_ids,attention_mask,)
        loss = loss + adv_alpha*loss_adv
        loss.backward()
        optimizer.step()
        model.zero_grad()
    """

    def __init__(self,
                 model,
                 epsilon=1e-6,
                 multi_gpu_on=False,
                 step_size=1e-3,
                 noise_var=1e-5,
                 norm_p='inf',
                 k=1,
                 fp16=False,
                 loss_map={},
                 norm_level=0):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.model = model
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0

    # 梯度scale
    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == 'l2':
            if sentence_level:
                direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon)
            else:
                direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon)
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction, eff_direction

    # 初始noise扰动
    def generate_noise(self, embed, mask, epsilon=1e-5):
        noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
        noise.detach()
        noise.requires_grad_()
        return noise

    # 对称散度loss
    def stable_kl(self, logit, target, epsilon=1e-6, reduce=True):
        logit = logit.view(-1, logit.size(-1)).float()
        target = target.view(-1, target.size(-1)).float()
        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
        if reduce:
            return (p * (rp - ry) * 2).sum() / bs
        else:
            return (p * (rp - ry) * 2).sum()

    # 对抗loss输出
    def forward(self,
                logits,
                input_ids,
                token_type_ids,
                attention_mask,
                task_id=0,
                task_type="Classification",
                pairwise=1):
        # adv training
        assert task_type in set(['Classification', 'Ranking', 'Regression']), 'Donot support {} yet'.format(task_type)
        vat_args = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        # init delta
        embed = self.model(**vat_args)  # embed [B,S,h_dim] h_dim=768
        # embed生成noise
        noise = self.generate_noise(embed, attention_mask, epsilon=self.noise_var)
        # noise更新K轮
        for step in range(0, self.K):
            vat_args = {'inputs_embeds': embed + noise}
            # noise+embed得到对抗样本的输出logits
            adv_logits = self.model(**vat_args)
            if task_type == 'Regression':
                adv_loss = F.mse_loss(adv_logits, logits.detach(), reduction='sum')
            else:
                if task_type == 'Ranking':
                    adv_logits = adv_logits.view(-1, pairwise)
                adv_loss = self.stable_kl(adv_logits, logits.detach(), reduce=False)

            # 得到noise的梯度
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            # 得到新的scale的noise
            noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()
        vat_args = {'inputs_embeds': embed + noise}
        adv_logits = self.model(**vat_args)
        if task_type == 'Ranking':
            adv_logits = adv_logits.view(-1, pairwise)
        adv_lc = self.loss_map[task_id]
        # 计算对抗样本的对抗损失
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        return adv_loss

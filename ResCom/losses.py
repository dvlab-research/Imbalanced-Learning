import torch
import torch.nn as nn
import numpy as np


class SiamBS_SPM(nn.Module):
    def __init__(self, cls_num_list, queue_size_per_cls, balsfx_n=0.0, temperature=1, con_weight=1.0, effective_num_beta=0.999):
        super(SiamBS_SPM, self).__init__()
        self.temperature = temperature
        self.queue_size_per_cls = queue_size_per_cls
        self.con_weight = con_weight
        self.balsfx_n = balsfx_n
        self.effective_num_beta = effective_num_beta

        self.criterion_cls = nn.CrossEntropyLoss()
        self.log_sfx = nn.LogSoftmax(dim=1)
        self.criterion_con = torch.nn.KLDivLoss(reduction='none')

        self._cal_prior_weight_for_classes(cls_num_list)

        self.queue_label = torch.repeat_interleave(torch.arange(len(cls_num_list)), self.queue_size_per_cls)
        self.instance_prior = self.class_prior.squeeze()[self.queue_label]

    def _cal_prior_weight_for_classes(self, cls_num_list):
        
        ### logit prior for BalSfx
        cls_num_list_tensor = torch.Tensor(cls_num_list).view(1, len(cls_num_list))
        self.class_prior = cls_num_list_tensor / cls_num_list_tensor.sum()
        self.class_prior = self.class_prior.to(torch.device('cuda'))
        
        # effective number on the orginal-batch level
        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))
            
        else:
            self.class_weight  = torch.FloatTensor(torch.ones(len(cls_num_list))).to(torch.device('cuda'))
        

    # Supervised contrastvie style
    def forward(self, sim_con, labels_con, logits_cls, labels):
        ### Siamese Balanced Softmax
        logits_cls = logits_cls + torch.log(self.class_prior + 1e-9)
        loss_cls = self.criterion_cls(logits_cls, labels)
        
        if self.balsfx_n == 0:
            sim_con = self.log_sfx(sim_con / self.temperature)
        else :
            instance_prior = torch.cat([self.class_prior.squeeze()[labels.squeeze()], self.instance_prior], dim=0)
            sim_con = self.log_sfx(sim_con / self.temperature + self.balsfx_n * torch.log(instance_prior + 1e-9))
        
        loss_con = self.criterion_con(sim_con, labels_con)
        loss_con = loss_con.sum(dim=1) / (labels_con.sum(dim=1) + 1e-9)

        instance_weight = self.class_weight.squeeze()[labels.squeeze()]
        loss_con = (instance_weight * loss_con).mean()

        # Total loss
        loss = loss_cls + self.con_weight * loss_con
        
        return loss_cls, loss_con, loss 
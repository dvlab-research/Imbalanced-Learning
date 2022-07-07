import torch
import torch.nn as nn


class ResCom(nn.Module):

    def __init__(self, base_encoder, dim_feat, batch_size, select_num_pos, select_num_neg, 
                 num_classes=1000, dim=128, queue_size_per_cls=8):
        """

        """
        super(ResCom, self).__init__()
        
        self.num_classes = num_classes
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = base_encoder(num_classes=dim)
        self.linear = nn.Linear(dim_feat, self.num_classes)
        self.mlp = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.BatchNorm1d(dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim))

        # create the queue
        self.queue_size_per_cls = queue_size_per_cls
        self.register_buffer("queue_list", torch.randn(dim, self.queue_size_per_cls * self.num_classes))
        self.queue_list = nn.functional.normalize(self.queue_list, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("queue_label", torch.repeat_interleave(torch.arange(self.num_classes), self.queue_size_per_cls))

        self.register_buffer("pos_index", torch.zeros(self.num_classes, self.queue_size_per_cls, dtype=torch.int64))
        self.register_buffer("neg_index", torch.zeros(self.num_classes, self.queue_size_per_cls * (self.num_classes - 1), dtype=torch.int64))
        
        self.register_buffer("ids", torch.arange(batch_size * 2))

        for i in range(self.num_classes):
            self.pos_index[i, :] = torch.arange(self.queue_size_per_cls) + i * self.queue_size_per_cls
            self.neg_index[i, :] = torch.cat([torch.arange(self.queue_size_per_cls * i), torch.arange(self.queue_size_per_cls * (i + 1), self.queue_size_per_cls * self.num_classes)], dim=0)

        self.select_num_pos = select_num_pos
        self.select_num_neg = select_num_neg
 

    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, feat_k, labels_k):
        
        feat_k_gather = concat_all_gather(feat_k)
        labels_k_gather = concat_all_gather(labels_k)
        
        # print(self.queue_ptr)
        
        for i in range(feat_k_gather.shape[0]):
            key_c = feat_k_gather[i:i+1, :]
            c = labels_k_gather[i]
            instance_size = 1
            ptr = int(self.queue_ptr[c])
            real_ptr = ptr + c * self.queue_size_per_cls
            # replace the keys at ptr (dequeue and enqueue)
            self.queue_list[:, real_ptr:real_ptr + instance_size] = key_c.T
            ptr = (ptr + instance_size) % self.queue_size_per_cls  # move pointer
            self.queue_ptr[c] = ptr
            
        # print(self.queue_ptr)



    def _train(self, img, labels):

        # compute query features
        mid_feat = self.encoder(img)  # queries: NxC

        feat = self.mlp(mid_feat)
        feat = nn.functional.normalize(feat, dim=1)

        logit_cls = self.linear(mid_feat)

        # compute contrast logits
        cur_queue_list = self.queue_list.clone().detach()
        # pos_samples = torch.gather(cur_queue_list, 1, self.pos_index[labels_q,:])
        sim_con_queue = feat @ cur_queue_list
        sim_con_pos = torch.gather(sim_con_queue, 1, self.pos_index[labels, :])
        sim_con_neg = torch.gather(sim_con_queue, 1, self.neg_index[labels, :])
        
        ### supervised hard positive and negative pairs mining
        # select hard pos (smaller)
        sim_con_pos_select, _ = torch.topk(-sim_con_pos, self.select_num_pos)
        sim_con_pos_select = -sim_con_pos_select
        # select hard neg (larger)
        sim_con_neg_select, _ = torch.topk(sim_con_neg, self.select_num_neg)
        

        sim_con_batch = feat @ feat.T
        mask = torch.ones_like(sim_con_batch).scatter_(1, self.ids.unsqueeze(1), 0.)
        sim_con_batch = sim_con_batch[mask.bool()].view(sim_con_batch.shape[0], sim_con_batch.shape[1] - 1)
        sim_con = torch.cat([sim_con_batch, sim_con_pos_select, sim_con_neg_select], dim=1)

        labels_con_batch = torch.eq(labels[:, None], labels[None, :]).float()
        labels_con_batch = labels_con_batch[mask.bool()].view(labels_con_batch.shape[0], labels_con_batch.shape[1] - 1)
        labels_con = torch.cat([labels_con_batch, torch.ones_like(sim_con_pos_select), torch.zeros_like(sim_con_neg_select)], dim=1)

        
        feat_k = feat[:feat.shape[0] // 2, :]
        labels_k = labels[:labels.shape[0] // 2]
    
        self._dequeue_and_enqueue(feat_k, labels_k)
                           
        return sim_con, labels_con, logit_cls

    def _inference(self, image):
        mid_feat = self.encoder(image)
        logit_cls = self.linear(mid_feat)
        return logit_cls

    def forward(self, img, labels=None):
        if self.training:
            return self._train(img, labels)  
        else:
           return self._inference(img)



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

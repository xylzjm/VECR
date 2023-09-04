import os.path as osp
import torch
import mmcv


class prototype_dist_estimator:
    def __init__(
        self,
        cfg,
        resume=None,
    ):
        super(prototype_dist_estimator, self).__init__()

        self.class_num = cfg['class_num']
        self.feat_num = cfg['feat_num']
        self.ignore_idx = cfg['ignore_idx']
        self.use_momentum = cfg['use_momentum']
        self.momentum = cfg['momentum']

        if resume is not None:
            mmcv.print_log(f'Loading prototype from {resume}', 'mmseg')
            ckpt = torch.load(resume, map_location='cpu')
            self.Proto = ckpt['Proto'].cuda(non_blocking=True)
            self.Amount = ckpt['Amount'].cuda(non_blocking=True)
        else:
            self.Proto = torch.zeros(self.class_num, self.feat_num).cuda(
                non_blocking=True
            )
            self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)

    def update(self, feat, label):
        mask = label != self.ignore_idx
        # remove IGNORE_LABEL pixels
        label = label[mask]
        feat = feat[mask]
        if not self.use_momentum:
            N, A = feat.shape
            C = self.class_num
            # refer to SDCA for fast implementation
            feat = feat.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, label.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            feat_by_sort = feat.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = feat_by_sort.sum(0) / Amount_CXA
            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(sum_weight + self.Amount.view(C, 1).expand(C, A))
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)
        else:
            # momentum implementation
            ids_unique = label.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = label == i
                feature = feat[mask_i]
                feature = torch.mean(feature, dim=0)
                self.Amount[i] += len(mask_i)
                self.Proto[i, :] = self.momentum * feature + self.Proto[i, :] * (
                    1 - self.momentum
                )

    def save(self, name):
        torch.save(
            {'Proto': self.Proto.cpu(), 'Amount': self.Amount.cpu()},
            osp.join('pretrained/', name),
        )

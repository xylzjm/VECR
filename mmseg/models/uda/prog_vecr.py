import os
import random

import matplotlib.pyplot as plt
import numpy as np

import mmcv
import torch
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.models import UDA
from mmseg.models.uda.vecr import VECR
from mmseg.models.utils.dacs_transforms import (
    denorm,
    get_class_masks,
    get_mean_std,
    strong_transform,
)
from mmseg.models.utils.fourier_transforms import fourier_transform
from mmseg.models.utils.night_fog_filter import night_fog_filter
from mmseg.models.utils.prototype_dist_estimator import prototype_dist_estimator
from mmseg.models.utils.visualization import subplotimg
from mmseg.ops import resize
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd


@UDA.register_module()
class ProG_VECR(VECR):
    def __init__(self, **cfg):
        super(ProG_VECR, self).__init__(**cfg)
        self.proto_cfg = cfg['proto']
        self.proto_resume = cfg['proto_resume']
        self.pseudo_threshold = None

        assert self.num_classes == self.proto_cfg['num_class']
        assert self.ignore_index == self.proto_cfg['ignore_index']
        self.feat_estimator = None

        assert cfg['stylize'].get('source', None)
        assert cfg['stylize'].get('target', None)
        self.stylization = cfg['stylize']

        self.src_inv_lambda = self.stylization['inv_loss']['weight']
        self.tgt_inv_lambda = self.stylization['inv_loss']['weight_target']
        mmcv.print_log(
            f'src_inv_lambda: {self.src_inv_lambda}, tgt_inv_lambda: {self.tgt_inv_lambda}',
            'mmseg',
        )

    def update_prototype_statistics(self, src_img, tgt_img, src_label, tgt_label):
        src_emafeat = self.get_ema_model().extract_bottlefeat(src_img)
        tgt_emafeat = self.get_ema_model().extract_bottlefeat(tgt_img)
        b, a, hs, ws = src_emafeat.shape
        _, _, ht, wt = tgt_emafeat.shape
        src_emafeat = (
            src_emafeat.permute(0, 2, 3, 1).contiguous().view(b * hs * ws, a)
        )
        tgt_emafeat = (
            tgt_emafeat.permute(0, 2, 3, 1).contiguous().view(b * ht * wt, a)
        )

        src_mask = (
            resize(src_label.float(), size=(hs, ws), mode='nearest')
            .long()
            .contiguous()
            # fmt: off
            .view(b * hs * ws, )
            # fmt: on
        )
        tgt_mask = (
            resize(tgt_label.float(), size=(ht, wt), mode='nearest')
            .long()
            .contiguous()
            # fmt: off
            .view(b * ht * wt, )
            # fmt: on
        )

        self.feat_estimator.update(feat=tgt_emafeat.detach(), label=tgt_mask)
        self.feat_estimator.update(feat=src_emafeat.detach(), label=src_mask)

    def estimate_pseudo_weight(self, proto, feat, pseudo_lbl):
        """
        Args:
            C means NUM_CLASS, A means feature dim.

            proto: shape of (C, A), the mean representation of each class.
            feat: shape of (B, A, H, W).
            pseudo_lbl: shape of (B, H, W)

            Return: pixel-wise pseudo confidence, which is shape of (B, H, W)
        """
        b, a, h, w = feat.shape
        c, _ = proto.shape
        assert pseudo_lbl.shape == (b, h, w)

        feat = feat.permute(0, 2, 3, 1).contiguous().view(b * h * w, a)
        feat = F.normalize(feat, p=2, dim=1)
        proto = F.normalize(proto, p=2, dim=1)

        weight = feat @ proto.permute(1, 0).contiguous()
        weight = weight.softmax(dim=1).view(b, h, w, c).permute(0, 3, 1, 2)

        pseudo_weight = weight.gather(1, pseudo_lbl.unsqueeze(1))

        return pseudo_weight.squeeze(1)

    def calculate_feat_invariance(self, f1, f2, proto, source):
        """
        Args:
            C means NUM_CLASS, A means feature dim.

            proto: shape of (C, A), the mean representation of each class.
            feat: shape of (B, A, H, W).

            Return: feature invariance loss.
        """
        assert f1.shape == f2.shape
        b, a, h, w = f1.shape

        f1 = f1.permute(0, 2, 3, 1).contiguous().view(b * h * w, a)
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = f2.permute(0, 2, 3, 1).contiguous().view(b * h * w, a)
        f2 = F.normalize(f2, p=2, dim=1)
        proto = F.normalize(proto, p=2, dim=1)

        f1_dis = f1 @ proto.permute(1, 0).contiguous()
        f2_dis = f2 @ proto.permute(1, 0).contiguous()

        item1 = F.kl_div(f1_dis.log_softmax(1), f2_dis.softmax(1), reduction='mean')
        item2 = F.kl_div(f2_dis.log_softmax(1), f1_dis.softmax(1), reduction='mean')
        item = (item1 + item2) / 2.0

        item *= self.src_inv_lambda if source else self.tgt_inv_lambda
        inv_loss, inv_log = self._parse_losses({'loss_feat_inv': item})
        inv_log.pop('loss', None)
        return inv_loss, inv_log

    def forward_train(self, img, img_metas, gt_semantic_seg, return_feat=False):
        log_vars = {}
        batch_size = img[0][0].shape[0]
        dev = img[0][0].device
        means, stds = get_mean_std(img_metas[0][0], dev)

        # Init/update ema model and prototype
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())
            self.feat_estimator = prototype_dist_estimator(
                self.proto_cfg, resume=self.proto_resume
            )

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        src_img, tgt_img = img[0][0], img[1][0]
        src_img_metas, tgt_img_metas = img_metas[0][0], img_metas[1][0]
        src_gt_semantic_seg, tgt_gt_semantic_seg = (
            gt_semantic_seg[0][0],
            gt_semantic_seg[1][0],
        )

        # illumination boost image
        night_map = []
        for meta in tgt_img_metas:
            if 'night' in meta['filename']:
                night_map.append(1)
            else:
                night_map.append(0)
        tgt_ib_img = night_fog_filter(
            tgt_img, means, stds, night_map, mode='hsv-s-w4'
        )
        # Fourier amplitude transform
        tgt_fb_img, src_fb_img = [None] * batch_size, [None] * batch_size
        for i in range(batch_size):
            tgt_fb_img[i] = fourier_transform(
                data=torch.stack((tgt_ib_img[i], src_img[i])),
                mean=means[0].unsqueeze(0),
                std=stds[0].unsqueeze(0),
            )
            src_fb_img[i] = fourier_transform(
                data=torch.stack((src_img[i], tgt_img[i])),
                mean=means[0].unsqueeze(0),
                std=stds[0].unsqueeze(0),
            )
        tgt_fb_img = torch.cat(tgt_fb_img)
        src_fb_img = torch.cat(src_fb_img)
        del tgt_ib_img

        # train main model with source
        if (
            self.stylization['source']['ce_original']
            or self.stylization['source']['consist']
        ):
            src_losses = self.get_model().forward_train(
                src_img,
                src_img_metas,
                src_gt_semantic_seg,
                get_bottlefeat=self.stylization['source']['consist'],
            )
            if self.stylization['source']['consist']:
                src_feat = src_losses.pop('bottlefeat')
            assert 'bottlefeat' not in src_losses
            if self.stylization['source']['ce_original']:
                src_loss, src_log_vars = self._parse_losses(src_losses)
                log_vars.update(add_prefix(src_log_vars, f'src'))
                src_loss.backward(retain_graph=self.stylization['source']['consist'])

        # train main model with styled source
        if (
            self.stylization['source']['ce_stylized']
            or self.stylization['source']['consist']
        ):
            src_style_losses = self.get_model().forward_train(
                src_fb_img,
                src_img_metas,
                src_gt_semantic_seg,
                get_bottlefeat=self.stylization['source']['consist'],
            )
            if self.stylization['source']['consist']:
                src_style_feat = src_style_losses.pop('bottlefeat')
            assert 'bottlefeat' not in src_style_losses
            if self.stylization['source']['ce_stylized']:
                # fmt: off
                src_style_loss, src_style_log_vars = self._parse_losses(src_style_losses)
                log_vars.update(add_prefix(src_style_log_vars, f'src_stylized'))
                src_style_loss.backward(retain_graph=self.stylization['source']['consist'])
                # fmt: on

        # feature invariance loss between original and stylized versions of source images.
        if self.stylization['source']['consist']:
            src_inv_loss, src_inv_log = self.calculate_feat_invariance(
                src_feat,
                src_style_feat,
                proto=self.feat_estimator.Proto.detach(),
                source=True,
            )
            log_vars.update(add_prefix(src_inv_log, 'src'))
            src_inv_loss.backward()
        try:
            del src_feat, src_style_feat
        except NameError:
            pass

        # generate target pseudo label from ema model
        tgt_outputs = self.get_ema_model().encode_decode_bottlefeat(
            tgt_fb_img, tgt_img_metas
        )
        tgtema_logits, tgtema_feat = tgt_outputs['out'], tgt_outputs['feat']
        tgt_softmax = torch.softmax(tgtema_logits.detach(), dim=1)
        pseudo_label = torch.argmax(tgt_softmax, dim=1)
        # update feature statistics
        tgt_ps_semantic_seg = pseudo_label.clone().unsqueeze(1)
        tgt_ps_semantic_seg[
            torch.ne(tgt_ps_semantic_seg, tgt_gt_semantic_seg)
        ] = self.ignore_index
        self.update_prototype_statistics(
            src_img, tgt_img, src_gt_semantic_seg, tgt_ps_semantic_seg
        )
        # estimate pseudo weight
        pseudo_weight = self.estimate_pseudo_weight(
            proto=self.feat_estimator.Proto.detach(),
            feat=tgtema_feat.detach(),
            pseudo_lbl=pseudo_label,
        )
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)
        del tgt_outputs, tgtema_logits, tgtema_feat, tgt_softmax

        # prepare for dacs transforms
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1) if self.color_jitter else 0,
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0),
        }

        # dacs mixed target
        mix_masks = get_class_masks(src_gt_semantic_seg)
        mixed_img = [None] * len(self.stylization['target']['ce'])
        mixed_lbl, feats_pool = [None] * batch_size, {}
        tgt_inv_flag = self.stylization['target']['consist'] is not None
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            _, mixed_lbl[i] = strong_transform(
                strong_parameters,
                target=torch.stack((src_gt_semantic_seg[i][0], pseudo_label[i])),
            )
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])),
            )
        mixed_lbl = torch.cat(mixed_lbl)
        for j, ce_args in enumerate(self.stylization['target']['ce']):
            mixed_img[j] = [None] * batch_size
            if ce_args[0] == 'original':
                src_stdby = src_img
            elif ce_args[0] == 'stylized':
                src_stdby = src_fb_img
            else:
                raise ValueError(f'{ce_args[0]} not allowed target CE argument')
            if ce_args[1] == 'original':
                tgt_stdby = tgt_img
            elif ce_args[1] == 'stylized':
                tgt_stdby = tgt_fb_img
            else:
                raise ValueError(f'{ce_args[1]} not allowed target CE argument')
            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[j][i], _ = strong_transform(
                    strong_parameters, data=torch.stack((src_stdby[i], tgt_stdby[i]))
                )
            mixed_img[j] = torch.cat(mixed_img[j])

            # train main model with target
            mix_losses = self.get_model().forward_train(
                mixed_img[j],
                tgt_img_metas,
                mixed_lbl,
                pseudo_weight,
                get_bottlefeat=tgt_inv_flag,
            )
            if tgt_inv_flag and ce_args in self.stylization['target']['consist']:
                feats_pool[ce_args] = mix_losses.pop('bottlefeat')
            assert 'bottlefeat' not in mix_losses
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(
                add_prefix(mix_log_vars, f'mix_{ce_args[0]}_{ce_args[1]}')
            )
            mix_loss.backward(
                retain_graph=(ce_args in self.stylization['target']['consist'])
            )

        # feature invariance loss between original and stylized versions of target images.
        if tgt_inv_flag:
            for con_args in self.stylization['target']['consist']:
                if con_args not in feats_pool:
                    # fmt: off
                    mixed_img_ = [None] * batch_size
                    if con_args[0] == 'original':
                        src_stdby = src_img
                    elif con_args[0] == 'stylized':
                        src_stdby = src_fb_img
                    else:
                        raise ValueError(f'{con_args[0]} not allowed target CE argument')
                    if con_args[1] == 'original':
                        tgt_stdby = tgt_img
                    elif con_args[1] == 'stylized':
                        tgt_stdby = tgt_fb_img
                    else:
                        raise ValueError(f'{con_args[1]} not allowed target CE argument')
                    for i in range(batch_size):
                        strong_parameters['mix'] = mix_masks[i]
                        mixed_img_[i], _ = strong_transform(
                            strong_parameters, data=torch.stack((src_stdby[i], tgt_stdby[i]))
                        )
                    mixed_img_ = torch.cat(mixed_img_)
                    feats_pool[con_args] = self.get_model().extract_bottlefeat(mixed_img_)
                    # fmt: on
            assert len(feats_pool) == len(self.stylization['target']['consist'])
            tgt_inv_loss, tgt_inv_log = self.calculate_feat_invariance(
                list(feats_pool.values())[0],
                list(feats_pool.values())[1],
                proto=self.feat_estimator.Proto.detach(),
                source=False,
            )
            log_vars.update(add_prefix(tgt_inv_log, 'tgt'))
            tgt_inv_loss.backward()
            del feats_pool

        # visualize
        if (
            self.debug_img_interval is not None
            and self.local_iter % self.debug_img_interval == 0
        ):
            out_dir = os.path.join(self.train_cfg['work_dir'], 'visualize_meta')
            os.makedirs(out_dir, exist_ok=True)
            vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
            vis_tgt_img = torch.clamp(denorm(tgt_img, means, stds), 0, 1)
            vis_mix_img = torch.clamp(denorm(mixed_img[0], means, stds), 0, 1)
            vis_src_fb_img = torch.clamp(denorm(src_fb_img, means, stds), 0, 1)
            vis_tgt_fb_img = torch.clamp(denorm(tgt_fb_img, means, stds), 0, 1)
            vis_mix_fb_img = torch.clamp(denorm(mixed_img_, means, stds), 0, 1)
            with torch.no_grad():
                # source pseudo label
                src_logits = self.get_model().encode_decode(src_img, src_img_metas)
                src_softmax = torch.softmax(src_logits.detach(), dim=1)
                _, src_pseudo_label = torch.max(src_softmax, dim=1)
                src_pseudo_label = src_pseudo_label.unsqueeze(1)
                # source ema label
                src_logits = self.get_ema_model().encode_decode(
                    src_img, src_img_metas
                )
                src_softmax = torch.softmax(src_logits.detach(), dim=1)
                _, src_ema_label = torch.max(src_softmax, dim=1)
                src_ema_label = src_ema_label.unsqueeze(1)
                # source fb label
                src_logits = self.get_model().encode_decode(
                    src_fb_img, src_img_metas
                )
                src_softmax = torch.softmax(src_logits.detach(), dim=1)
                _, src_fb_label = torch.max(src_softmax, dim=1)
                src_fb_label = src_fb_label.unsqueeze(1)
                # target pseudo label
                tgt_logits = self.get_model().encode_decode(tgt_img, tgt_img_metas)
                tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
                _, tgt_pseudo_label = torch.max(tgt_softmax, dim=1)
                tgt_pseudo_label = tgt_pseudo_label.unsqueeze(1)
                # target fb label
                tgt_logits = self.get_model().encode_decode(
                    tgt_fb_img, tgt_img_metas
                )
                tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
                _, tgt_fb_label = torch.max(tgt_softmax, dim=1)
                tgt_fb_label = tgt_fb_label.unsqueeze(1)
                # target ema label
                tgt_logits = self.get_ema_model().encode_decode(
                    tgt_fb_img, tgt_img_metas
                )
                tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
                _, tgt_ema_fb_label = torch.max(tgt_softmax, dim=1)
                tgt_ema_fb_label = tgt_ema_fb_label.unsqueeze(1)
                # mixed label pred
                mix_logits = self.get_model().encode_decode(
                    mixed_img[0], tgt_img_metas
                )
                mix_softmax = torch.softmax(mix_logits.detach(), dim=1)
                _, mix_label_test = torch.max(mix_softmax, dim=1)
                mix_label_test = mix_label_test.unsqueeze(1)
                # mixed fb label pred
                mix_logits = self.get_model().encode_decode(
                    mixed_img_, tgt_img_metas
                )
                mix_softmax = torch.softmax(mix_logits.detach(), dim=1)
                _, mix_fb_label_test = torch.max(mix_softmax, dim=1)
                mix_fb_label_test = mix_fb_label_test.unsqueeze(1)

            for j in range(batch_size):
                rows, cols = 3, 6
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0,
                    },
                )
                # source visualization
                subplotimg(
                    axs[0][0],
                    vis_src_img[j],
                    f'{os.path.basename(src_img_metas[j]["filename"])}',
                )
                subplotimg(
                    axs[0][1],
                    vis_src_fb_img[j],
                    f'Source FB',
                )
                subplotimg(
                    axs[0][2],
                    src_gt_semantic_seg[j],
                    f'Source GT',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                subplotimg(
                    axs[0][3],
                    src_pseudo_label[j],
                    f'Source PL',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                subplotimg(
                    axs[0][4],
                    src_ema_label[j],
                    f'Source EMA PL',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                subplotimg(
                    axs[0][5],
                    src_fb_label[j],
                    f'Source FB TEST',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                # target visualization
                subplotimg(
                    axs[1][0],
                    vis_tgt_img[j],
                    f'{os.path.basename(tgt_img_metas[j]["filename"])}',
                )
                subplotimg(axs[1][1], vis_tgt_fb_img[j], f'Target FB')
                subplotimg(
                    axs[1][2],
                    tgt_gt_semantic_seg[j],
                    f'Target GT',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                subplotimg(
                    axs[1][3],
                    tgt_pseudo_label[j],
                    f'Target PL',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                subplotimg(
                    axs[1][4],
                    tgt_ema_fb_label[j],
                    f'Target EMA FB PL',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                subplotimg(
                    axs[1][5],
                    tgt_fb_label[j],
                    f'Target FB TEST',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                # mixed visualization
                subplotimg(axs[2][0], vis_mix_img[j], f'Mixed')
                subplotimg(axs[2][1], vis_mix_fb_img[j], f'Mixed FB')
                subplotimg(
                    axs[2][2], pseudo_weight[j], 'Pseudo Weight', vmin=0, vmax=1
                )
                subplotimg(
                    axs[2][3],
                    mixed_lbl[j],
                    f'Mixed PL',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                subplotimg(
                    axs[2][4],
                    mix_label_test[j],
                    f'Mixed TEST',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )
                subplotimg(
                    axs[2][5],
                    mix_fb_label_test[j],
                    f'Mixed FB TEST',
                    cmap='cityscapes',
                    nc=self.num_classes,
                )

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png')
                )
                plt.close()
        self.local_iter += 1

        return log_vars

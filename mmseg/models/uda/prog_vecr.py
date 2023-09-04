import os
import random

import matplotlib.pyplot as plt
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
        self.feat_estimator = None
        self.pseudo_threshold = None

    def update_prototype_statistics(self, src_img, tgt_img, src_label, tgt_label):
        src_emafeat = self.get_ema_model().extract_bottlefeat(src_img)
        tgt_emafeat = self.get_ema_model().extract_bottlefeat(tgt_img)
        b, a, hs, ws = src_emafeat.shape
        _, _, ht, wt = tgt_emafeat.shape
        src_emafeat = src_emafeat.permute(0, 2, 3, 1).contiguous().view(b * hs * ws, a)
        tgt_emafeat = tgt_emafeat.permute(0, 2, 3, 1).contiguous().view(b * ht * wt, a)

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

    def calculate_pseudo_weight(self, Proto, feat, pseudo_label):
        """
        Args:
            C means NUM_CLASS, A means feature dim.
            Proto: shape of (C, A), the mean representation of each class.
            feat: shape of (B, A, H, W).
            pseudo_label: shape of (B, H, W).

            Return: pixel-wise pseudo weight, which is shape of (B, H, W)
        """
        b, a, h, w = feat.shape
        c, _ = Proto.shape
        assert pseudo_label.shape == (b, h, w)

        feat = feat.permute(0, 2, 3, 1).contiguous().view(b * h * w, a)
        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)

        weight = feat @ Proto.permute(1, 0).contiguous()
        weight = weight.softmax(dim=1)

        pseudo_weight = weight.view(b, h, w, c).permute(0, 3, 1, 2)
        pseudo_weight = pseudo_weight.gather(1, pseudo_label.unsqueeze(1))

        return pseudo_weight.squeeze(1)

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
        tgt_ib_img = night_fog_filter(tgt_img, means, stds, night_map, mode='hsv-s-w4')
        # Fourier amplitude transform
        tgt_fb_img = [None] * batch_size
        for i in range(batch_size):
            tgt_fb_img[i] = fourier_transform(
                data=torch.stack((tgt_ib_img[i], src_img[i])),
                mean=means[0].unsqueeze(0),
                std=stds[0].unsqueeze(0),
            )
        tgt_fb_img = torch.cat(tgt_fb_img)
        del tgt_ib_img

        # train main model with source
        src_losses = self.get_model().forward_train(
            src_img, src_img_metas, src_gt_semantic_seg, return_feat=False
        )
        src_loss, src_log_vars = self._parse_losses(src_losses)
        log_vars.update(add_prefix(src_log_vars, f'src'))
        src_loss.backward()

        # update feature statistics
        self.update_prototype_statistics(
            src_img, tgt_img, src_gt_semantic_seg, tgt_gt_semantic_seg
        )

        # generate target pseudo label from ema model
        tgt_outputs = self.get_ema_model().encode_decode_bottlefeat(
            tgt_fb_img, tgt_img_metas
        )
        tgt_feat, tgt_logits = tgt_outputs['feat'], tgt_outputs['out']
        tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
        pseudo_label = torch.argmax(tgt_softmax, dim=1)
        pseudo_weight = self.calculate_pseudo_weight(
            Proto=self.feat_estimator.Proto.detach(),
            feat=tgt_feat,
            pseudo_label=pseudo_label,
        )
        # mmcv.print_log(f'pseudo_weight shape: {pseudo_weight.shape}', 'mmseg')
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)
        del tgt_outputs, tgt_feat, tgt_logits, tgt_softmax

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
        mixed_img, mixed_lbl, mixed_fb_img = (
            [None] * batch_size,
            [None] * batch_size,
            [None] * batch_size,
        )
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((src_img[i], tgt_img[i])),
                target=torch.stack((src_gt_semantic_seg[i][0], pseudo_label[i])),
            )
            mixed_fb_img[i], pseudo_weight[i] = strong_transform(
                strong_parameters,
                data=torch.stack((src_img[i], tgt_fb_img[i])),
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])),
            )
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        mixed_fb_img = torch.cat(mixed_fb_img)

        # train main model with target
        mix_losses = self.get_model().forward_train(
            mixed_img, tgt_img_metas, mixed_lbl, pseudo_weight, return_feat=False
        )
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(add_prefix(mix_log_vars, 'mix'))
        mix_loss.backward()

        # train main model with target fb
        mix_fb_losses = self.get_model().forward_train(
            mixed_fb_img, tgt_img_metas, mixed_lbl, pseudo_weight, return_feat=False
        )
        mix_fb_loss, mix_fb_log_vars = self._parse_losses(mix_fb_losses)
        log_vars.update(add_prefix(mix_fb_log_vars, 'mix_fb'))
        mix_fb_loss.backward()

        # visualize
        if (
            self.debug_img_interval is not None
            and self.local_iter % self.debug_img_interval == 0
        ):
            out_dir = os.path.join(self.train_cfg['work_dir'], 'visualize_meta')
            os.makedirs(out_dir, exist_ok=True)
            vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
            vis_tgt_img = torch.clamp(denorm(tgt_img, means, stds), 0, 1)
            vis_mix_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            vis_tgt_fb_img = torch.clamp(denorm(tgt_fb_img, means, stds), 0, 1)
            vis_mix_fb_img = torch.clamp(denorm(mixed_fb_img, means, stds), 0, 1)
            with torch.no_grad():
                # source pseudo label
                src_logits = self.get_model().encode_decode(src_img, src_img_metas)
                src_softmax = torch.softmax(src_logits.detach(), dim=1)
                _, src_pseudo_label = torch.max(src_softmax, dim=1)
                src_pseudo_label = src_pseudo_label.unsqueeze(1)
                # source ema label
                src_logits = self.get_ema_model().encode_decode(src_img, src_img_metas)
                src_softmax = torch.softmax(src_logits.detach(), dim=1)
                _, src_ema_label = torch.max(src_softmax, dim=1)
                src_ema_label = src_ema_label.unsqueeze(1)
                # target pseudo label
                tgt_logits = self.get_model().encode_decode(tgt_img, tgt_img_metas)
                tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
                _, tgt_pseudo_label = torch.max(tgt_softmax, dim=1)
                tgt_pseudo_label = tgt_pseudo_label.unsqueeze(1)
                # target fb label
                tgt_logits = self.get_model().encode_decode(tgt_fb_img, tgt_img_metas)
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
                mix_logits = self.get_model().encode_decode(mixed_img, tgt_img_metas)
                mix_softmax = torch.softmax(mix_logits.detach(), dim=1)
                _, mix_label_test = torch.max(mix_softmax, dim=1)
                mix_label_test = mix_label_test.unsqueeze(1)
                # mixed fb label pred
                mix_logits = self.get_model().encode_decode(mixed_fb_img, tgt_img_metas)
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
                subplotimg(axs[2][2], pseudo_weight[j], 'Pseudo Weight', vmin=0, vmax=1)
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

import argparse
import mmcv
import torch
import torch.nn.functional as F

from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models.builder import build_segmentor
from mmcv.runner import load_checkpoint
from mmcv.parallel import scatter
from mmcv.parallel import MMDataParallel
from mmseg.models.utils.prototype_dist_estimator import prototype_dist_estimator


def prototype_initialize(model, data_loader, device, cfg):
    mmcv.print_log(f'---------------- Initialize prototype ----------------', 'mmseg')
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    feat_estimator = prototype_dist_estimator(cfg.proto, resume=None)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = scatter(data, device)[0]
            src_img, src_label = data['img'], data['gt_semantic_seg']

            src_feat = model.module.extract_bottlefeat(src_img)
            B, N, Hs, Ws = src_feat.shape

            # source mask: downsample the ground-truth label
            src_mask = (
                F.interpolate(src_label.float(), size=(Hs, Ws), mode='nearest')
                .squeeze(0)
                .long()
            )
            src_mask = src_mask.contiguous().view(
                B * Hs * Ws,
            )

            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)
            feat_estimator.update(feat=src_feat.detach().clone(), label=src_mask)

            for _ in range(B):
                prog_bar.update()
        mmcv.print_log('')
        feat_estimator.save('prototype_initial_value.pth')


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Target Prototype and initialize"
    )
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set random seeds
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        mmcv.print_log(f'Set random seed to {args.seed}', 'mmseg')
        set_random_seed(args.seed, deterministic=False)
    cfg.seed = args.seed
    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # build the dataloader
    data_cfg = cfg.data.train
    mmcv.print_log(f'data_pipeline: {data_cfg["pipeline"]}', 'mmseg')
    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        1,
        cfg.data.workers_per_gpu,
        dist=False,
        seed=cfg.seed,
        drop_last=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')
    )
    load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')],
    )
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    prototype_initialize(model, data_loader, cfg.gpu_ids, cfg)


if __name__ == '__main__':
    main()

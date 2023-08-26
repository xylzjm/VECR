# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis.inference import LoadImage
from mmseg.core.evaluation import get_classes, get_palette
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_train_model


def init_model(
    config,
    checkpoint=None,
    device='cuda:0',
    classes=None,
    palette=None,
):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            'config must be a filename or Config object, '
            'but got {}'.format(type(config))
        )
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_train_model(config, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES'] if classes is None \
            else classes
        model.PALETTE = checkpoint['meta']['PALETTE'] if palette is None \
            else palette
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def variance_forward(model, img):
    """Forward prediction variance with the model.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Tensor: The prediction variance map.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    with torch.no_grad():
        pred_s = model.get_model().encode_decode(data['img'][0], data['img_metas'])
        pred_t = model.get_ema_model().encode_decode(data['img'][0], data['img_metas'])
    var = torch.sum(
        kl_distance(
            torch.log_softmax(pred_t, dim=1),
            torch.softmax(pred_s, dim=1),
        ),
        dim=1,
    )
    exp_var = torch.exp(-var)
    mmcv.print_log(f'var shape: {var.shape}, exp_var shape: {exp_var.shape}', 'mmseg')
    mmcv.print_log(f'exp_var: {exp_var}')
    return var


def plotimg(
    img,
    out_file=None,
    **kwargs,
):
    if img is None:
        return
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()

    plt.imsave(out_file, img, **kwargs)


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map',
    )
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale']
    )
    model = init_model(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
    )
    # test a single image
    var_map = variance_forward(model, args.img)
    # show the results
    file, extension = os.path.splitext(args.img)
    var_file = f'{file}_variance{extension}'
    assert var_file != args.img
    plotimg(var_map, var_file, vmin=0, vmax=1)
    print('Save variance map to', var_file)


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import mmcv

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette, get_classes


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
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.',
    )
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale']
    )
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')],
    )
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    file, extension = os.path.splitext(args.img)
    pred_file = f'{file}_pred{extension}'
    assert pred_file != args.img
    model.show_result(
        args.img,
        result,
        palette=get_palette(args.palette),
        out_file=pred_file,
        show=False,
        opacity=args.opacity)
    print('Save prediction to', pred_file)


if __name__ == '__main__':
    main()

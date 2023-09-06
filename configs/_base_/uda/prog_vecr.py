uda = dict(
    type='ProG_VECR',
    alpha=0.999,
    pseudo_threshold=0.9,
    debug_img_interval=1000,
    proto=dict(
        # prototype estimator
        feat_num=256*4,
        ignore_index=255,
        use_momentum=False,
        momentum=0.9),
    stylize=dict(
        source=dict(
            ce_original=True,
            ce_stylized=False,
            consist=True,
        ),
        target=dict(
            ce=[('original', 'original')],
            consist=[('original', 'original'), ('stylized', 'stylized')],
        )),
    proto_resume='pretrained/prototype_source_initial_value.pth'
)
use_ddp_wrapper = True

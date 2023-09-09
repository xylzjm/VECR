uda = dict(
    type='ProG_VECR',
    alpha=0.999,
    pseudo_threshold=0.9,
    debug_img_interval=1000,
    proto=dict(
        # prototype estimator
        feat_dim=256*4,
        num_class=19,
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
        ),
        inv_loss=dict(
            # Feature invariance loss weights set to 50 for source Cityscapes domain and 20 for target ACDC domain.
            weight=50.0,
            weight_target=20.0
        )),
    proto_resume='pretrained/prototype_source_initial_value.pth'
)
use_ddp_wrapper = True

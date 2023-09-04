uda = dict(
    type='PROG_VECR',
    alpha=0.999,
    pseudo_threshold=0.9,
    debug_img_interval=1000,
    proto=dict(
        # prototype estimator
        feat_num=256*4,
        class_num=19,
        ignore_index=255,
        use_momentum=False,
        momentum=0.9),
    proto_resume='pretrained/prototype_source_initial_value.pth'
)
use_ddp_wrapper = True

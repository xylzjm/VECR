_base_ = ['prog_vecr.py']

uda = dict(
    stylize=dict(
        source=dict(
            ce_original=True,
            ce_stylized=True,
            consist=False,
        ),
        target=dict(
            ce=[('original', 'original'), ('stylized', 'stylized')],
            consist=None
        )))
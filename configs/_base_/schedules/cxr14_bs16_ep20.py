# optimizer
optimizer = dict(type='Adam', lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15, 18])
runner = dict(type='EpochBasedRunner', max_epochs=20)
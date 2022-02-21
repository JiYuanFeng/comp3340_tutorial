_base_ = [
    '_base_/datasets/cxr14_bs32.py',
    '_base_/models/resnet50_cxr14.py',
    '_base_/schedules/cxr14_bs32_ep5.py',
    '_base_/default_runtime.py'
]
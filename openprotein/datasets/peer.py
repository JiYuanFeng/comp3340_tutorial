# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from torchdrug.datasets import Fluorescence, Stability, BetaLactamase, Solubility
from torchdrug.datasets import SubcellularLocalization, BinaryLocalization
from torchdrug.datasets import Fold, SecondaryStructure
from torchdrug.datasets import PPIAffinity, HumanPPI, YeastPPI
from torchdrug.datasets import PDBBind, BindingDB

from torch.utils.data import Dataset
from .base_dataset import expanduser
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class Peer(Dataset):

    def __init__(self,
                 data_prefix,
                 pipeline,
                 ann_file=None,
                 test_mode=False):
        super(Dataset, self).__init__()
        self.data_prefix = expanduser(data_prefix)
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        pass

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)


@DATASETS.register_module()
class BetaLactamase(BetaLactamase):
    def __init__(self, split, **kwargs):
        if not isinstance(split, list):
            self.splits = [split]
        super(BetaLactamase, self).__init__(**kwargs)
    # def load_annotations(self):
    #     pass
    #
    # def prepare_data(self, idx):
    #     results = copy.deepcopy(self.data_infos[idx])
    #     return self.pipeline(results)
    #
    # def __len__(self):
    #     return len(self.data_infos)
    #
    # def __getitem__(self, idx):
    #     return self.prepare_data(idx)
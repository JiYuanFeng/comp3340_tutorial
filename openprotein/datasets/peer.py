# Copyright (c) OpenMMLab. All rights reserved.
import os
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
from torchdrug import data, utils
from .base_dataset import expanduser
from .builder import DATASETS
from .pipelines import Compose
from .protein import ProteinDataset


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
class BetaLactamase(ProteinDataset):
    """
    The activity values of first-order mutants of the TEM-1 beta-lactamase protein.

    Statistics:
        - #Train: 4,158
        - #Valid: 520
        - #Test: 520

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/beta_lactamase.tar.gz"
    md5 = "65766a3969cc0e94b101d4063d204ba4"
    target_fields = ["scaled_effect1"]

    def __init__(self, path, split="train", verbose=1, test_mode=False, **kwargs):
        self.splits = [split]
        self.test_mode = test_mode

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "beta_lactamase/beta_lactamase_%s.lmdb" % split)
                      for split in self.splits]

        self.load_lmdbs(lmdb_files, target_fields=self.target_fields, verbose=verbose, **kwargs)

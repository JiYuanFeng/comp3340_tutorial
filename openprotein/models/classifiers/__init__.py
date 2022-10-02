# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .graph import GraphClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', "GraphClassifier"]

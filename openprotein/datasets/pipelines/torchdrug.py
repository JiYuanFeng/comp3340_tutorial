from torchdrug.transforms import ProteinView
from ..builder import PIPELINES


PIPELINES.register_module(module=ProteinView)
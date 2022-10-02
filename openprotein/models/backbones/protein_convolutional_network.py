from ..builder import BACKBONES
from torchdrug.models import ProteinConvolutionalNetwork

BACKBONES.register_module(ProteinConvolutionalNetwork)
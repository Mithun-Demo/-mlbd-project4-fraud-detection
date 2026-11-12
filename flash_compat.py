"""Compatibility shims so Lightning Flash 0.8.2 (unmaintained) imports against Lightning 2.x."""
import sys, types
from lightning_utilities.core.apply_func import apply_to_collection
m = types.ModuleType("pytorch_lightning.utilities.apply_func")
m.apply_to_collection = apply_to_collection
sys.modules["pytorch_lightning.utilities.apply_func"] = m

# pytorch-tabular 1.0.2 (the version Flash 0.8.2 targets) builds pytorch_tabnet.TabNet without a group
# attention matrix; pytorch-tabnet 4.x then receives the old default [] and crashes. An identity matrix is
# the documented "no feature grouping" value, so behaviour is unchanged.
import torch
from pytorch_tabnet import tab_network

_orig_init = tab_network.TabNet.__init__


def _patched_init(self, input_dim, output_dim, *args, **kwargs):
    if kwargs.get("group_attention_matrix", []) == []:
        kwargs["group_attention_matrix"] = torch.eye(input_dim)
    _orig_init(self, input_dim, output_dim, *args, **kwargs)


tab_network.TabNet.__init__ = _patched_init

"""Restore kornia names that the third-party submodules still use but kornia 0.9 removed.

The submodules are pinned upstream repos that we do not edit, so instead of patching them we re-register the
removed names, each pointing at its current kornia location. Imported for its side effects by ``vismatch``.
"""

import sys
from types import ModuleType

import torch
import kornia
import kornia.color
import kornia.core
import kornia.core.utils
import kornia.geometry
import kornia.utils


def _add_module(name: str, **attrs) -> None:
    """Register ``name`` as a module holding ``attrs`` and bind it on its parent, unless kornia still ships it."""
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    setattr(sys.modules[parent], child, module)


# kornia.utils.grid was removed in 0.8.3 (LoFTR-family fine matching: aspanformer, eloftr, matchanything,
# matchformer, se2loftr, topicfm, minima-loftr).
_add_module(
    "kornia.utils.grid",
    create_meshgrid=kornia.geometry.create_meshgrid,
    create_meshgrid3d=kornia.geometry.create_meshgrid3d,
)
# kornia.utils.helpers was removed (MatchAnything's hloc adalam matcher).
_add_module(
    "kornia.utils.helpers",
    get_cuda_device_if_available=kornia.core.utils.get_cuda_device_if_available,
)
# kornia.core no longer re-exports torch's Module/Tensor (aspanformer).
for _name, _value in {"Module": torch.nn.Module, "Tensor": torch.Tensor}.items():
    if not hasattr(kornia.core, _name):
        setattr(kornia.core, _name, _value)
# kornia.rgb_to_grayscale top-level alias was removed (imatch-toolbox doghardnet-nn).
if not hasattr(kornia, "rgb_to_grayscale"):
    kornia.rgb_to_grayscale = kornia.color.rgb_to_grayscale

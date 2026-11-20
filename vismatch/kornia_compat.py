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
import kornia.geometry
import kornia.utils

# kornia.utils.grid was removed in 0.8.3 (LoFTR-family fine matching: aspanformer, eloftr, matchanything,
# matchformer, se2loftr, topicfm, minima-loftr).
if "kornia.utils.grid" not in sys.modules:
    kornia.utils.grid = sys.modules["kornia.utils.grid"] = ModuleType("kornia.utils.grid")
    kornia.utils.grid.create_meshgrid = kornia.geometry.create_meshgrid
# kornia.core no longer re-exports torch's Module/Tensor (aspanformer).
if not hasattr(kornia.core, "Module"):
    kornia.core.Module = torch.nn.Module
if not hasattr(kornia.core, "Tensor"):
    kornia.core.Tensor = torch.Tensor
# kornia.rgb_to_grayscale top-level alias was removed (imatch-toolbox doghardnet-nn).
if not hasattr(kornia, "rgb_to_grayscale"):
    kornia.rgb_to_grayscale = kornia.color.rgb_to_grayscale

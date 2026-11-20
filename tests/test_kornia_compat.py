"""The kornia names that third-party submodules still use resolve once vismatch is imported (no weights needed)."""

import importlib

import pytest
import torch

import vismatch  # noqa: F401 - installs vismatch.kornia_compat


@pytest.mark.parametrize(
    ("module", "name", "expected"),
    [
        ("kornia.utils.grid", "create_meshgrid", "kornia.geometry.create_meshgrid"),
        ("kornia.core", "Module", "torch.nn.Module"),
        ("kornia.core", "Tensor", "torch.Tensor"),
        ("kornia", "rgb_to_grayscale", "kornia.color.rgb_to_grayscale"),
    ],
)
def test_legacy_kornia_name(module, name, expected):
    """``from <module> import <name>`` works and gives the object at its current location."""
    expected_module, _, expected_name = expected.rpartition(".")
    obj = getattr(importlib.import_module(module), name)
    assert obj is getattr(importlib.import_module(expected_module), expected_name)


def test_legacy_create_meshgrid_runs():
    """The LoFTR-family fine matchers call kornia.utils.grid.create_meshgrid(h, w, normalized, device)."""
    from kornia.utils.grid import create_meshgrid

    grid = create_meshgrid(3, 4, False, torch.device("cpu"))
    assert grid.shape == (1, 3, 4, 2)

"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2025-01-18
"""

import torch
from typing import List


"""
Basic utilities related to manteinance and execution of PyTorch models.
"""


def find_unused_parameters(model: torch.nn.Module) -> List[str]:
    """
    Detect unusued parameters within a PyTorch module.

    IMPORTANT: This function must be called between model.backward() and optimizer.step()

    Args:
        model: torch.nn.Module

    Returns:
        unused_param_names: the str names of unused parameters.
    """
    unused_params = []

    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)

    return unused_params
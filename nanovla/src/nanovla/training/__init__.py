"""
nanoVLA Training utilities including RL training.
"""

from typing import TYPE_CHECKING

from ..utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["rl_training"] = [
        "nanoVLA_RL_Trainer",
        "RLTrainingConfig", 
        "create_rl_training_config",
    ]

if TYPE_CHECKING:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .rl_training import RLTrainingConfig, create_rl_training_config, nanoVLA_RL_Trainer

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

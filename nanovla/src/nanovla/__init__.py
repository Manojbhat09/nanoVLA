# Copyright 2024 The nanoVLA Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import nanovla` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "0.1.0"

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
    is_diffusers_available,
    is_vision_available,
)

# Base imports that are always available
_import_structure = {
    "configuration_utils": ["VLAConfig", "PretrainedConfig"],
    "utils": [
        "logging",
        "is_torch_available",
        "is_transformers_available", 
        "is_diffusers_available",
        "is_vision_available",
    ],
}

# Core modeling utilities (always available if torch is installed)
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects  # noqa F401

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["modeling_utils"] = [
        "BaseVLM",
        "PretrainedVLMMixin", 
        "CompatibilityMixin",
        "ModuleUtilsMixin",
    ]
    _import_structure["models"] = [
        "nanoVLA",
        "VLABuilder",
        "VLAFactory",
    ]
    _import_structure["models.components"] = [
        "VisionEncoder",
        "LanguageModel", 
        "CrossModalFusion",
        "ActionDecoder",
    ]

# VLM integrations (requires transformers)
try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_transformers_objects  # noqa F401

    _import_structure["utils.dummy_transformers_objects"] = [
        name for name in dir(dummy_transformers_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.vlms"] = [
        "LLaVAVLM",
        "BlipVLM", 
        "InstructBLIPVLM",
        "OpenVLAVLM",
    ]
    _import_structure["registry"] = [
        "vlm_registry",
        "VLMCapabilities",
        "CompatibilityLevel",
        "TransferLearningStrategy",
    ]

# Training utilities
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["training"] = [
        "VLATrainer",
        "TrainingArguments",
        "VLATrainingArguments",
    ]
    _import_structure["evaluation"] = [
        "VLMBenchmark",
        "evaluate_model",
        "compute_metrics",
    ]

# Pipeline integrations
try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["pipelines"] = [
        "VLAPipeline",
        "ActionPredictionPipeline",
        "pipeline",
    ]

# Vision utilities
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_vision_objects  # noqa F401

    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing"] = [
        "VLAImageProcessor",
        "BaseImageProcessor",
    ]

# Direct imports for type checking
if TYPE_CHECKING:
    # Configuration
    from .configuration_utils import PretrainedConfig, VLAConfig
    from .utils import (
        OptionalDependencyNotAvailable,
        is_diffusers_available,
        is_torch_available,
        is_transformers_available,
        is_vision_available,
        logging,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # Core modeling
        from .modeling_utils import BaseVLM, CompatibilityMixin, ModuleUtilsMixin, PretrainedVLMMixin
        from .models import VLABuilder, VLAFactory, nanoVLA
        from .models.components import ActionDecoder, CrossModalFusion, LanguageModel, VisionEncoder

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # VLM integrations
        from .models.vlms import BlipVLM, InstructBLIPVLM, LLaVAVLM, OpenVLAVLM
        from .registry import CompatibilityLevel, TransferLearningStrategy, VLMCapabilities, vlm_registry

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # Training
        from .evaluation import VLMBenchmark, compute_metrics, evaluate_model
        from .training import TrainingArguments, VLATrainer, VLATrainingArguments

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # Pipelines
        from .pipelines import ActionPredictionPipeline, VLAPipeline, pipeline

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # Vision processing
        from .image_processing import BaseImageProcessor, VLAImageProcessor

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)


# Convenient shortcuts following transformers pattern
def __getattr__(name: str):
    """Provide convenient access to main classes"""
    if name == "nanoVLA":
        from .models import nanoVLA as _nanoVLA
        return _nanoVLA
    elif name == "VLABuilder":
        from .models import VLABuilder as _VLABuilder
        return _VLABuilder
    elif name == "vlm_registry":
        from .registry import vlm_registry as _vlm_registry
        return _vlm_registry
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")





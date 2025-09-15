# Copyright 2024 The nanoVLA Team. All rights reserved.
# Adapted from HuggingFace Transformers
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

"""
Import utilities for nanoVLA.
"""

import functools
import importlib
import importlib.metadata
import importlib.util
import inspect
import operator as op
import os
import sys
import time
from typing import Any, Dict, Optional, Union

from packaging import version


# The package importlib_metadata is in a different place, depending on the Python version.
try:
    importlib_metadata = importlib.metadata
except AttributeError:
    import importlib_metadata


logger = None  # Will be set when logging is imported


class OptionalDependencyNotAvailable(Exception):
    """Exception raised when an optional dependency is not available."""


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[bool, str]:
    """
    Check if a package is available.
    
    Args:
        pkg_name: Name of the package to check
        return_version: Whether to return version string instead of bool
        
    Returns:
        Boolean indicating availability, or version string if return_version=True
    """
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib_metadata.version(pkg_name)
            package_exists = True
        except importlib_metadata.PackageNotFoundError:
            package_exists = False
            if logger is not None:
                logger.debug(f"Detected {pkg_name}, but unable to retrieve version")
    if return_version:
        return package_version if package_exists else "N/A"
    else:
        return package_exists


def _compare_versions(op_func, current_version: str, expected_version: str) -> bool:
    """
    Compare two version strings using the provided operator.
    
    Args:
        op_func: Operator function (e.g., op.ge for >=)
        current_version: Current version string
        expected_version: Expected version string
        
    Returns:
        Result of version comparison
    """
    try:
        return op_func(version.parse(current_version), version.parse(expected_version))
    except Exception:
        return False


# PyTorch
_torch_available = _is_package_available("torch")


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    return _torch_available


def is_torch_version(op_func, torch_version: str) -> bool:
    """Check if PyTorch version satisfies the given condition."""
    if not _torch_available:
        return False
    current_version = _is_package_available("torch", return_version=True)
    return _compare_versions(op_func, current_version, torch_version)


# Transformers
_transformers_available = _is_package_available("transformers")


def is_transformers_available() -> bool:
    """Check if transformers is available."""
    return _transformers_available


def is_transformers_version(op_func, transformers_version: str) -> bool:
    """Check if transformers version satisfies the given condition."""
    if not _transformers_available:
        return False
    current_version = _is_package_available("transformers", return_version=True)
    return _compare_versions(op_func, current_version, transformers_version)


# Diffusers
_diffusers_available = _is_package_available("diffusers")


def is_diffusers_available() -> bool:
    """Check if diffusers is available.""" 
    return _diffusers_available


def is_diffusers_version(op_func, diffusers_version: str) -> bool:
    """Check if diffusers version satisfies the given condition."""
    if not _diffusers_available:
        return False
    current_version = _is_package_available("diffusers", return_version=True)
    return _compare_versions(op_func, current_version, diffusers_version)


# Vision dependencies
_pillow_available = _is_package_available("PIL")
_opencv_available = _is_package_available("cv2")
_torchvision_available = _is_package_available("torchvision")


def is_vision_available() -> bool:
    """Check if basic vision dependencies are available."""
    return _pillow_available and _torchvision_available


def is_opencv_available() -> bool:
    """Check if OpenCV is available."""
    return _opencv_available


# Robotics dependencies
_gymnasium_available = _is_package_available("gymnasium")


def is_robotics_available() -> bool:
    """Check if robotics dependencies are available."""
    return _gymnasium_available


# Training dependencies
_wandb_available = _is_package_available("wandb")
_tensorboard_available = _is_package_available("tensorboard")
_datasets_available = _is_package_available("datasets")
_accelerate_available = _is_package_available("accelerate")


def is_training_available() -> bool:
    """Check if training dependencies are available."""
    return _datasets_available and _accelerate_available


def is_wandb_available() -> bool:
    """Check if Weights & Biases is available."""
    return _wandb_available


def is_tensorboard_available() -> bool:
    """Check if TensorBoard is available."""
    return _tensorboard_available


# Quality dependencies
_ruff_available = _is_package_available("ruff")
_black_available = _is_package_available("black")
_mypy_available = _is_package_available("mypy")


def is_quality_available() -> bool:
    """Check if code quality tools are available."""
    return _ruff_available and _black_available


# Lazy module implementation adapted from transformers
class _LazyModule(sys.modules[__name__].__class__):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    def __init__(self, name: str, module_file: str, import_structure: Dict[str, Any], module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(self._class_to_module.keys())
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements that are not already there.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self._name} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self._name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self._name}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


def check_min_version(package_name: str, min_version: str) -> None:
    """
    Check that a package version is at least the minimum required.
    
    Args:
        package_name: Name of the package
        min_version: Minimum required version
        
    Raises:
        ImportError: If package is not available or version is too old
    """
    if not _is_package_available(package_name):
        raise ImportError(f"Package {package_name} is required but not installed.")
    
    current_version = _is_package_available(package_name, return_version=True)
    if not _compare_versions(op.ge, current_version, min_version):
        raise ImportError(
            f"Package {package_name} version {current_version} is too old. "
            f"Minimum required version is {min_version}."
        )


def get_available_devices() -> list[str]:
    """Get list of available compute devices."""
    devices = ["cpu"]
    
    if is_torch_available():
        import torch
        
        if torch.cuda.is_available():
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
    
    return devices


def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """Get GPU memory information if available."""
    if not (is_torch_available() and is_torch_available()):
        return None
    
    import torch
    
    if not torch.cuda.is_available():
        return None
    
    device_count = torch.cuda.device_count()
    memory_info = {}
    
    for i in range(device_count):
        memory_info[f"cuda:{i}"] = {
            "total": torch.cuda.get_device_properties(i).total_memory,
            "allocated": torch.cuda.memory_allocated(i),
            "cached": torch.cuda.memory_reserved(i),
        }
    
    return memory_info

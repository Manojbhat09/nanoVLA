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

"""
Utilities for nanoVLA library, following transformers patterns.
"""

import sys

from .import_utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_diffusers_available,
    is_torch_available,
    is_transformers_available,
    is_vision_available,
)
from . import logging


# Dummy object utilities for missing backends
class DummyObject(type):
    """
    Metaclass for dummy objects. Any class that uses DummyObject as metaclass will return a dummy object that will
    throw an error if used.
    """

    def __call__(cls, *args, **kwargs):
        return cls._raise_error_with_class(cls.__name__)

    def __getattr__(cls, key):
        return cls._raise_error_with_class(cls.__name__)

    def _raise_error_with_class(cls, class_name):
        raise ImportError(
            f"Using `{class_name}` requires the installation of the following: {cls._backends}. "
            f"Please install the missing dependencies by running `pip install -U "
            f"{' '.join([bckend for bckend in cls._backends])}` and restart your Python runtime."
        )


def requires_backends(obj, backends):
    """
    Checks if the object is used with the required backends. If not, throws an ImportError with a helpful message.
    """
    name = obj.__class__.__name__
    if not all(getattr(sys.modules[__name__.split(".")[0]], f"is_{backend}_available")() for backend in backends):
        raise ImportError(f"{name} requires {backends}. See https://nanovla.readthedocs.io/installation for installation instructions.")


__all__ = [
    "OptionalDependencyNotAvailable",
    "_LazyModule", 
    "is_torch_available",
    "is_transformers_available",
    "is_diffusers_available", 
    "is_vision_available",
    "logging",
    "DummyObject",
    "requires_backends",
]

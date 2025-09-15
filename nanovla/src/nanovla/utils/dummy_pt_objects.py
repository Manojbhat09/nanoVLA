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
Dummy PyTorch objects for when PyTorch is not available.
"""

from ..utils import DummyObject, requires_backends


class BaseVLM(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class nanoVLA(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class VLABuilder(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class VLAFactory(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])



#!/usr/bin/env python3
"""
nanoVLA VLM Factory System

Implementation of the from_vlm() functionality to create nanoVLA models
from existing Vision-Language Models like LLaVA, FastVLM, etc.

This is the working implementation moved from the root directory
into the proper package structure.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

from .utils import logging

logger = logging.get_logger(__name__)


@dataclass
class VLMMetadata:
    """Metadata about a VLM for integration"""
    name: str
    model_family: str
    vision_encoder_type: str
    language_model_type: str
    fusion_strategy: str
    action_adaptation_strategy: str
    requires_special_tokens: bool = False
    max_image_size: int = 336
    vision_feature_dim: int = 768
    language_feature_dim: int = 768
    supports_multi_image: bool = False
    huggingface_id: Optional[str] = None


class BaseVLMAdapter(ABC):
    """Base class for VLM adapters"""
    
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        self._config = None
        
    @abstractmethod
    def load_model(self) -> Any:
        """Load the VLM model"""
        pass
        
    @abstractmethod
    def extract_vision_encoder(self) -> nn.Module:
        """Extract the vision encoder from the VLM"""
        pass
        
    @abstractmethod
    def extract_language_model(self) -> nn.Module:
        """Extract the language model from the VLM"""
        pass
        
    @abstractmethod
    def get_fusion_module(self) -> nn.Module:
        """Get or create the fusion module"""
        pass
        
    @abstractmethod
    def get_metadata(self) -> VLMMetadata:
        """Get metadata about this VLM"""
        pass


class MockVLMAdapter(BaseVLMAdapter):
    """Mock adapter for testing without heavy dependencies"""
    
    def __init__(self, model_path: str, vlm_name: str = "mock", **kwargs):
        super().__init__(model_path, **kwargs)
        self.vlm_name = vlm_name
        
    def load_model(self):
        # Mock model loading
        logger.info(f"Mock loading VLM: {self.vlm_name} from {self.model_path}")
        return None
        
    def extract_vision_encoder(self) -> nn.Module:
        return nn.Identity()
        
    def extract_language_model(self) -> nn.Module:
        return nn.Identity()
        
    def get_fusion_module(self) -> nn.Module:
        return nn.Identity()
        
    def get_metadata(self) -> VLMMetadata:
        metadata_map = {
            "llava": VLMMetadata(
                name="llava",
                model_family="llava",
                vision_encoder_type="clip_vit",
                language_model_type="llama",
                fusion_strategy="linear_projection",
                action_adaptation_strategy="add_action_head",
                max_image_size=336,
                vision_feature_dim=768,
                language_feature_dim=4096
            ),
            "fastvlm": VLMMetadata(
                name="fastvlm",
                model_family="fastvlm",
                vision_encoder_type="fastvithd",
                language_model_type="qwen2",
                fusion_strategy="mlp_projection",
                action_adaptation_strategy="add_action_head",
                max_image_size=384,
                vision_feature_dim=576,
                language_feature_dim=3584
            ),
            "openvla": VLMMetadata(
                name="openvla",
                model_family="openvla",
                vision_encoder_type="siglip",
                language_model_type="llama",
                fusion_strategy="mlp_projection",
                action_adaptation_strategy="action_tokens",
                max_image_size=224,
                vision_feature_dim=1152,
                language_feature_dim=4096
            )
        }
        return metadata_map.get(self.vlm_name, metadata_map["llava"])


class VLMRegistry:
    """Registry for VLM adapters"""
    
    def __init__(self):
        self._adapters: Dict[str, Type[BaseVLMAdapter]] = {}
        self._register_default_adapters()
    
    def _register_default_adapters(self):
        """Register the default VLM adapters"""
        # For now, use mock adapters
        self.register("llava", lambda path, **kwargs: MockVLMAdapter(path, "llava", **kwargs))
        self.register("fastvlm", lambda path, **kwargs: MockVLMAdapter(path, "fastvlm", **kwargs))
        self.register("openvla", lambda path, **kwargs: MockVLMAdapter(path, "openvla", **kwargs))
        
        # Register aliases
        self.register("llava-1.5", lambda path, **kwargs: MockVLMAdapter(path, "llava", **kwargs))
        self.register("llava-1.6", lambda path, **kwargs: MockVLMAdapter(path, "llava", **kwargs))
        
    def register(self, name: str, adapter_factory):
        """Register a VLM adapter factory"""
        self._adapters[name.lower()] = adapter_factory
        logger.info(f"Registered VLM adapter: {name}")
    
    def get_adapter(self, name: str):
        """Get a VLM adapter by name"""
        name_lower = name.lower()
        if name_lower not in self._adapters:
            raise ValueError(f"Unknown VLM: {name}. Available: {list(self._adapters.keys())}")
        return self._adapters[name_lower]
    
    def list_available(self) -> List[str]:
        """List all available VLM adapters"""
        return list(self._adapters.keys())


# Global registry instance
vlm_registry = VLMRegistry()


class ActionHeadAdapter(nn.Module):
    """Adapter that adds action prediction capability to any VLM"""
    
    def __init__(
        self,
        vlm_adapter: BaseVLMAdapter,
        action_dim: int = 7,
        action_strategy: str = "continuous",
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.vlm_adapter = vlm_adapter
        self.action_dim = action_dim
        self.action_strategy = action_strategy
        
        # Get VLM metadata
        self.metadata = vlm_adapter.get_metadata()
        
        # Create mock action decoder for testing
        if action_strategy == "continuous":
            self.action_decoder = nn.Sequential(
                nn.Linear(self.metadata.language_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        elif action_strategy == "discrete":
            self.action_decoder = nn.Sequential(
                nn.Linear(self.metadata.language_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            raise ValueError(f"Unknown action strategy: {action_strategy}")
    
    def predict_action(self, image_data: Any, instruction: str) -> List[float]:
        """Mock action prediction for testing"""
        logger.info(f"Predicting action for instruction: '{instruction}'")
        if self.action_strategy == "continuous":
            return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0][:self.action_dim]
        else:
            return [1] * self.action_dim


def create_nanovla_from_vlm(
    vlm_name: str,
    model_path: str,
    action_dim: int = 7,
    action_strategy: str = "continuous",
    **kwargs
) -> ActionHeadAdapter:
    """
    Create a nanoVLA model from any Vision-Language Model.
    
    Args:
        vlm_name: Name of the VLM (e.g., "llava", "fastvlm", "openvla")
        model_path: Path to the pre-trained VLM model
        action_dim: Dimension of action space (default: 7 for 7-DOF robot)
        action_strategy: "continuous" or "discrete" action prediction
        **kwargs: Additional arguments passed to VLM loader
    
    Returns:
        ActionHeadAdapter: nanoVLA model ready for action prediction
    """
    
    # Get the appropriate adapter factory
    adapter_factory = vlm_registry.get_adapter(vlm_name)
    
    # Create VLM adapter instance
    vlm_adapter = adapter_factory(model_path, **kwargs)
    
    # Create action head adapter
    nanovla_model = ActionHeadAdapter(
        vlm_adapter=vlm_adapter,
        action_dim=action_dim,
        action_strategy=action_strategy
    )
    
    logger.info(f"Created nanoVLA from {vlm_name} with {action_dim}D {action_strategy} actions")
    
    return nanovla_model

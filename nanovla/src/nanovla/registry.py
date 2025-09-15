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
Advanced registry system for nanoVLA components.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Generic
import logging

from .utils import logging as nanovla_logging


logger = nanovla_logging.get_logger(__name__)

T = TypeVar('T')


class CompatibilityLevel(Enum):
    """Levels of VLM compatibility with VLA tasks"""
    NATIVE = auto()        # Designed for VLA (e.g., OpenVLA)
    COMPATIBLE = auto()    # Easy adaptation (e.g., LLaVA, InstructBLIP)
    ADAPTABLE = auto()     # Requires modification (e.g., CLIP + GPT)
    INCOMPATIBLE = auto()  # Cannot be used directly


class TransferLearningStrategy(Enum):
    """Different strategies for adapting VLMs to VLA tasks"""
    FREEZE_BACKBONE = "freeze_backbone"           # Freeze VLM, train only action head
    FINE_TUNE_ALL = "fine_tune_all"              # Fine-tune entire model
    PROGRESSIVE_UNFREEZING = "progressive"        # Gradually unfreeze layers
    ADAPTER_LAYERS = "adapters"                   # Use adapter/LoRA layers
    DISTILLATION = "distillation"                # Knowledge distillation
    HYBRID = "hybrid"                             # Combination of strategies


@dataclass
class VLMCapabilities:
    """Metadata about VLM capabilities"""
    supports_vision: bool
    supports_language: bool
    supports_multimodal_fusion: bool
    pretrained_on_robot_data: bool
    max_image_resolution: tuple[int, int]
    max_sequence_length: int
    feature_dimensions: dict[str, int]
    compatibility_level: CompatibilityLevel
    transfer_learning_strategy: str
    
    # Additional metadata
    model_family: str = "unknown"
    paper_url: Optional[str] = None
    huggingface_id: Optional[str] = None
    requirements: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


class ComponentRegistry(Generic[T]):
    """
    Advanced registry pattern with type safety and metadata
    """
    
    def __init__(self, component_type: str):
        self.component_type = component_type
        self._registry: Dict[str, Type[T]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._compatibility_cache: Dict[str, CompatibilityLevel] = {}
    
    def register(
        self, 
        name: str, 
        *, 
        capabilities: Optional[VLMCapabilities] = None,
        requirements: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator for registering components with metadata
        
        Example:
            @vlm_registry.register("my_vlm", capabilities=VLMCapabilities(...))
            class MyVLM(BaseVLM):
                pass
        """
        def decorator(cls: Type[T]) -> Type[T]:
            # Validate that class implements required protocol
            self._registry[name] = cls
            self._metadata[name] = {
                'capabilities': capabilities,
                'requirements': requirements or {},
                'aliases': aliases or [],
                'module': cls.__module__,
                'class_name': cls.__name__
            }
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    self._registry[alias] = cls
                    
            logger.info(f"Registered {self.component_type}: {name} -> {cls.__name__}")
            return cls
        
        return decorator
    
    def get(self, name: str) -> Optional[Type[T]]:
        """Get component class by name"""
        return self._registry.get(name)
    
    def create(self, name: str, *args, **kwargs) -> T:
        """Create instance of registered component"""
        cls = self.get(name)
        if cls is None:
            raise ValueError(f"Unknown {self.component_type}: {name}")
        return cls(*args, **kwargs)
    
    def list_available(self) -> List[str]:
        """List all registered component names"""
        # Return unique names (excluding aliases that point to same class)
        unique_names = []
        seen_classes = set()
        for name, cls in self._registry.items():
            if cls not in seen_classes:
                unique_names.append(name)
                seen_classes.add(cls)
        return unique_names
    
    def get_capabilities(self, name: str) -> Optional[VLMCapabilities]:
        """Get capabilities metadata for a component"""
        metadata = self._metadata.get(name, {})
        return metadata.get('capabilities')
    
    def get_requirements(self, name: str) -> Dict[str, Any]:
        """Get requirements for a component"""
        metadata = self._metadata.get(name, {})
        return metadata.get('requirements', {})
    
    def get_aliases(self, name: str) -> List[str]:
        """Get aliases for a component"""
        metadata = self._metadata.get(name, {})
        return metadata.get('aliases', [])
    
    @lru_cache(maxsize=128)
    def check_compatibility(self, vlm_name: str, action_dim: int) -> CompatibilityLevel:
        """Check compatibility between VLM and action requirements"""
        capabilities = self.get_capabilities(vlm_name)
        if capabilities is None:
            return CompatibilityLevel.INCOMPATIBLE
        
        # Implement sophisticated compatibility checking logic
        if capabilities.pretrained_on_robot_data:
            return CompatibilityLevel.NATIVE
        elif capabilities.supports_multimodal_fusion:
            return CompatibilityLevel.COMPATIBLE
        elif capabilities.supports_vision and capabilities.supports_language:
            return CompatibilityLevel.ADAPTABLE
        else:
            return CompatibilityLevel.INCOMPATIBLE
    
    def search_by_capability(self, **criteria) -> List[str]:
        """Search for VLMs by capabilities"""
        matches = []
        for name, metadata in self._metadata.items():
            capabilities = metadata.get('capabilities')
            if capabilities is None:
                continue
                
            match = True
            for key, value in criteria.items():
                if not hasattr(capabilities, key):
                    match = False
                    break
                if getattr(capabilities, key) != value:
                    match = False
                    break
            
            if match:
                matches.append(name)
        
        return matches
    
    def get_by_compatibility(self, level: CompatibilityLevel) -> List[str]:
        """Get all VLMs with specific compatibility level"""
        return self.search_by_capability(compatibility_level=level)
    
    def get_info_table(self) -> str:
        """Get formatted table of all registered components"""
        if not self._registry:
            return f"No {self.component_type}s registered."
        
        lines = [f"Registered {self.component_type}s:"]
        lines.append("=" * 80)
        lines.append(f"{'Name':<20} {'Class':<25} {'Compatibility':<15} {'Strategy':<15}")
        lines.append("-" * 80)
        
        for name in self.list_available():
            cls = self._registry[name]
            capabilities = self.get_capabilities(name)
            
            compat = capabilities.compatibility_level.name if capabilities else "UNKNOWN"
            strategy = capabilities.transfer_learning_strategy if capabilities else "unknown"
            
            lines.append(f"{name:<20} {cls.__name__:<25} {compat:<15} {strategy:<15}")
        
        return "\n".join(lines)


# Global registries
vlm_registry = ComponentRegistry[Any]("VLM")
action_decoder_registry = ComponentRegistry[Any]("ActionDecoder")
fusion_registry = ComponentRegistry[Any]("FusionModule")


# Registry helper functions
def register_vlm(name: str, **kwargs):
    """Convenience function to register VLMs"""
    return vlm_registry.register(name, **kwargs)


def register_action_decoder(name: str, **kwargs):
    """Convenience function to register action decoders"""
    return action_decoder_registry.register(name, **kwargs)


def register_fusion_module(name: str, **kwargs):
    """Convenience function to register fusion modules"""
    return fusion_registry.register(name, **kwargs)


def list_available_vlms() -> List[str]:
    """List all available VLMs"""
    return vlm_registry.list_available()


def get_vlm_info(name: str) -> Dict[str, Any]:
    """Get complete information about a VLM"""
    capabilities = vlm_registry.get_capabilities(name)
    requirements = vlm_registry.get_requirements(name)
    aliases = vlm_registry.get_aliases(name)
    
    return {
        "name": name,
        "capabilities": capabilities,
        "requirements": requirements,
        "aliases": aliases,
        "class": vlm_registry.get(name).__name__ if vlm_registry.get(name) else None
    }


def find_compatible_vlms(action_dim: int, min_compatibility: CompatibilityLevel = CompatibilityLevel.ADAPTABLE) -> List[str]:
    """Find VLMs compatible with given action dimension"""
    compatible = []
    for vlm_name in vlm_registry.list_available():
        compatibility = vlm_registry.check_compatibility(vlm_name, action_dim)
        if compatibility.value <= min_compatibility.value:  # Lower enum values = higher compatibility
            compatible.append(vlm_name)
    return compatible


def recommend_vlm_for_task(
    task_type: str = "robot_manipulation",
    action_dim: int = 7,
    performance_priority: str = "balanced"  # "speed", "accuracy", "balanced"
) -> List[str]:
    """Recommend VLMs for specific task requirements"""
    # Get compatible VLMs
    compatible = find_compatible_vlms(action_dim)
    
    if not compatible:
        return []
    
    # Score VLMs based on criteria
    scored_vlms = []
    for vlm_name in compatible:
        capabilities = vlm_registry.get_capabilities(vlm_name)
        if capabilities is None:
            continue
            
        score = 0
        
        # Base compatibility score
        if capabilities.compatibility_level == CompatibilityLevel.NATIVE:
            score += 10
        elif capabilities.compatibility_level == CompatibilityLevel.COMPATIBLE:
            score += 7
        elif capabilities.compatibility_level == CompatibilityLevel.ADAPTABLE:
            score += 4
        
        # Robot data pretraining bonus
        if capabilities.pretrained_on_robot_data:
            score += 5
        
        # Multimodal fusion bonus
        if capabilities.supports_multimodal_fusion:
            score += 3
        
        # Performance priority adjustments
        if performance_priority == "speed":
            # Favor smaller models and simpler architectures
            if "small" in vlm_name.lower() or "mini" in vlm_name.lower():
                score += 2
        elif performance_priority == "accuracy":
            # Favor larger, more sophisticated models
            if "large" in vlm_name.lower() or capabilities.feature_dimensions.get("vision", 0) > 1024:
                score += 2
        
        scored_vlms.append((vlm_name, score))
    
    # Sort by score and return names
    scored_vlms.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored_vlms]


def validate_vlm_requirements(vlm_name: str) -> Dict[str, bool]:
    """Validate that all requirements for a VLM are met"""
    requirements = vlm_registry.get_requirements(vlm_name)
    validation_results = {}
    
    for req_name, req_version in requirements.items():
        try:
            if req_name == "transformers":
                from .utils.import_utils import is_transformers_available, is_transformers_version
                import operator
                validation_results[req_name] = is_transformers_available() and is_transformers_version(operator.ge, req_version)
            elif req_name == "torch":
                from .utils.import_utils import is_torch_available, is_torch_version
                import operator
                validation_results[req_name] = is_torch_available() and is_torch_version(operator.ge, req_version)
            else:
                # Generic package check
                try:
                    import importlib
                    importlib.import_module(req_name)
                    validation_results[req_name] = True
                except ImportError:
                    validation_results[req_name] = False
        except Exception:
            validation_results[req_name] = False
    
    return validation_results


# Export main components
__all__ = [
    "CompatibilityLevel",
    "TransferLearningStrategy", 
    "VLMCapabilities",
    "ComponentRegistry",
    "vlm_registry",
    "action_decoder_registry",
    "fusion_registry",
    "register_vlm",
    "register_action_decoder", 
    "register_fusion_module",
    "list_available_vlms",
    "get_vlm_info",
    "find_compatible_vlms",
    "recommend_vlm_for_task",
    "validate_vlm_requirements",
]





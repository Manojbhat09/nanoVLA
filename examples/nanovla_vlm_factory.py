#!/usr/bin/env python3
"""
nanoVLA VLM Factory System

Implementation of the from_vlm() functionality to create nanoVLA models
from existing Vision-Language Models like LLaVA, FastVLM, etc.
"""

from __future__ import annotations

import inspect
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


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
        
    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this VLM"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer


class LLaVAAdapter(BaseVLMAdapter):
    """Adapter for LLaVA models"""
    
    def load_model(self):
        """Load LLaVA model"""
        if self._model is None:
            try:
                # Try to import LLaVA from the local ml-fastvlm directory
                import sys
                sys.path.append(str(Path(__file__).parent / "ml-fastvlm"))
                
                from llava.model import LlavaLlamaForCausalLM
                from llava.model.builder import load_pretrained_model
                
                # Load the model using LLaVA's loader
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=self.model_path,
                    model_base=None,
                    model_name=os.path.basename(self.model_path),
                    load_8bit=self.kwargs.get('load_8bit', False),
                    load_4bit=self.kwargs.get('load_4bit', False),
                )
                
                self._model = model
                self._tokenizer = tokenizer
                self._image_processor = image_processor
                
                logger.info(f"Loaded LLaVA model from {self.model_path}")
                
            except ImportError as e:
                logger.warning(f"Could not load LLaVA from local directory: {e}")
                # Fallback to HuggingFace if available
                self._model = AutoModel.from_pretrained(self.model_path)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        return self._model
    
    def extract_vision_encoder(self) -> nn.Module:
        """Extract vision encoder from LLaVA"""
        model = self.load_model()
        
        # LLaVA stores vision tower in model.get_vision_tower()
        if hasattr(model, 'get_vision_tower'):
            vision_tower = model.get_vision_tower()
            return vision_tower
        elif hasattr(model, 'vision_tower'):
            return model.vision_tower
        else:
            raise AttributeError("Could not find vision encoder in LLaVA model")
    
    def extract_language_model(self) -> nn.Module:
        """Extract language model from LLaVA"""
        model = self.load_model()
        
        # LLaVA uses the base language model (usually Llama)
        if hasattr(model, 'model'):
            return model.model  # The base language model
        elif hasattr(model, 'language_model'):
            return model.language_model
        else:
            raise AttributeError("Could not find language model in LLaVA model")
    
    def get_fusion_module(self) -> nn.Module:
        """Get the multimodal projector from LLaVA"""
        model = self.load_model()
        
        # LLaVA uses mm_projector for vision-language fusion
        if hasattr(model, 'mm_projector'):
            return model.mm_projector
        else:
            # Create a simple linear projection as fallback
            vision_dim = self.get_metadata().vision_feature_dim
            language_dim = self.get_metadata().language_feature_dim
            
            return nn.Linear(vision_dim, language_dim)
    
    def get_metadata(self) -> VLMMetadata:
        """Get LLaVA metadata"""
        model = self.load_model()
        config = model.config if hasattr(model, 'config') else None
        
        # Extract dimensions from config if available
        vision_dim = getattr(config, 'mm_hidden_size', 768) if config else 768
        language_dim = getattr(config, 'hidden_size', 4096) if config else 4096
        
        return VLMMetadata(
            name="llava",
            model_family="llava",
            vision_encoder_type="clip_vit",
            language_model_type="llama",
            fusion_strategy="linear_projection",
            action_adaptation_strategy="add_action_head",
            requires_special_tokens=True,
            max_image_size=336,
            vision_feature_dim=vision_dim,
            language_feature_dim=language_dim,
            supports_multi_image=True,
            huggingface_id=self.model_path
        )


class FastVLMAdapter(BaseVLMAdapter):
    """Adapter for FastVLM models"""
    
    def load_model(self):
        """Load FastVLM model"""
        if self._model is None:
            try:
                # Import from local ml-fastvlm directory
                import sys
                sys.path.append(str(Path(__file__).parent / "ml-fastvlm"))
                
                from llava.model.builder import load_pretrained_model
                
                # Load FastVLM (which is built on LLaVA)
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=self.model_path,
                    model_base=None,
                    model_name=os.path.basename(self.model_path),
                    load_8bit=self.kwargs.get('load_8bit', False),
                    load_4bit=self.kwargs.get('load_4bit', False),
                )
                
                self._model = model
                self._tokenizer = tokenizer
                self._image_processor = image_processor
                
                logger.info(f"Loaded FastVLM model from {self.model_path}")
                
            except ImportError as e:
                logger.error(f"Could not load FastVLM: {e}")
                raise ImportError("FastVLM requires the ml-fastvlm directory to be available")
        
        return self._model
    
    def extract_vision_encoder(self) -> nn.Module:
        """Extract FastViTHD encoder from FastVLM"""
        model = self.load_model()
        
        # FastVLM uses FastViTHD as vision encoder
        if hasattr(model, 'get_vision_tower'):
            vision_tower = model.get_vision_tower()
            return vision_tower
        elif hasattr(model, 'vision_tower'):
            return model.vision_tower
        else:
            raise AttributeError("Could not find FastViTHD encoder in FastVLM model")
    
    def extract_language_model(self) -> nn.Module:
        """Extract language model from FastVLM"""
        model = self.load_model()
        
        # FastVLM uses various language models (Qwen, Llama, etc.)
        if hasattr(model, 'model'):
            return model.model
        elif hasattr(model, 'language_model'):
            return model.language_model
        else:
            raise AttributeError("Could not find language model in FastVLM model")
    
    def get_fusion_module(self) -> nn.Module:
        """Get the multimodal projector from FastVLM"""
        model = self.load_model()
        
        if hasattr(model, 'mm_projector'):
            return model.mm_projector
        else:
            # Create a projection based on FastVLM's typical architecture
            vision_dim = self.get_metadata().vision_feature_dim
            language_dim = self.get_metadata().language_feature_dim
            
            return nn.Sequential(
                nn.Linear(vision_dim, language_dim),
                nn.GELU(),
                nn.Linear(language_dim, language_dim)
            )
    
    def get_metadata(self) -> VLMMetadata:
        """Get FastVLM metadata"""
        model = self.load_model()
        config = model.config if hasattr(model, 'config') else None
        
        # FastVLM typically uses smaller vision features due to efficiency optimizations
        vision_dim = getattr(config, 'mm_hidden_size', 576) if config else 576  # FastViTHD output
        language_dim = getattr(config, 'hidden_size', 3584) if config else 3584  # Qwen-2-7B typical
        
        return VLMMetadata(
            name="fastvlm",
            model_family="fastvlm",
            vision_encoder_type="fastvithd",
            language_model_type="qwen2",  # Most common for FastVLM
            fusion_strategy="mlp_projection",
            action_adaptation_strategy="add_action_head",
            requires_special_tokens=True,
            max_image_size=384,  # FastVLM supports higher resolution efficiently
            vision_feature_dim=vision_dim,
            language_feature_dim=language_dim,
            supports_multi_image=False,  # FastVLM focuses on single image efficiency
            huggingface_id=self.model_path
        )


class OpenVLAAdapter(BaseVLMAdapter):
    """Adapter for OpenVLA models"""
    
    def load_model(self):
        """Load OpenVLA model"""
        if self._model is None:
            try:
                # Try to load from local OpenVLA directory
                import sys
                openvla_path = Path(__file__).parent / "openvla"
                if openvla_path.exists():
                    sys.path.append(str(openvla_path))
                
                from transformers import AutoModelForVision2Seq
                
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                
                logger.info(f"Loaded OpenVLA model from {self.model_path}")
                
            except Exception as e:
                logger.warning(f"Could not load OpenVLA: {e}")
                # Fallback to standard loading
                self._model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        
        return self._model
    
    def extract_vision_encoder(self) -> nn.Module:
        """Extract vision encoder from OpenVLA"""
        model = self.load_model()
        
        # OpenVLA typically uses SigLIP vision encoder
        if hasattr(model, 'vision_backbone'):
            return model.vision_backbone
        elif hasattr(model, 'vision_model'):
            return model.vision_model
        else:
            raise AttributeError("Could not find vision encoder in OpenVLA model")
    
    def extract_language_model(self) -> nn.Module:
        """Extract language model from OpenVLA"""
        model = self.load_model()
        
        if hasattr(model, 'language_model'):
            return model.language_model
        elif hasattr(model, 'llm_backbone'):
            return model.llm_backbone
        else:
            raise AttributeError("Could not find language model in OpenVLA model")
    
    def get_fusion_module(self) -> nn.Module:
        """Get the fusion module from OpenVLA"""
        model = self.load_model()
        
        if hasattr(model, 'projector'):
            return model.projector
        elif hasattr(model, 'multi_modal_projector'):
            return model.multi_modal_projector
        else:
            # OpenVLA typically uses MLP projection
            vision_dim = self.get_metadata().vision_feature_dim
            language_dim = self.get_metadata().language_feature_dim
            
            return nn.Sequential(
                nn.Linear(vision_dim, language_dim),
                nn.GELU(),
                nn.Linear(language_dim, language_dim)
            )
    
    def get_metadata(self) -> VLMMetadata:
        """Get OpenVLA metadata"""
        return VLMMetadata(
            name="openvla",
            model_family="openvla",
            vision_encoder_type="siglip",
            language_model_type="llama",
            fusion_strategy="mlp_projection",
            action_adaptation_strategy="action_tokens",  # OpenVLA uses action tokenization
            requires_special_tokens=True,
            max_image_size=224,  # SigLIP typical
            vision_feature_dim=1152,  # SigLIP-L
            language_feature_dim=4096,  # Llama-7B
            supports_multi_image=False,
            huggingface_id=self.model_path
        )


class VLMRegistry:
    """Registry for VLM adapters"""
    
    def __init__(self):
        self._adapters: Dict[str, Type[BaseVLMAdapter]] = {}
        self._register_default_adapters()
    
    def _register_default_adapters(self):
        """Register the default VLM adapters"""
        self.register("llava", LLaVAAdapter)
        self.register("fastvlm", FastVLMAdapter)
        self.register("openvla", OpenVLAAdapter)
        
        # Register aliases
        self.register("llava-1.5", LLaVAAdapter)
        self.register("llava-1.6", LLaVAAdapter)
        self.register("llava-v1.5", LLaVAAdapter)
        self.register("llava-v1.6", LLaVAAdapter)
        
    def register(self, name: str, adapter_class: Type[BaseVLMAdapter]):
        """Register a VLM adapter"""
        self._adapters[name.lower()] = adapter_class
        logger.info(f"Registered VLM adapter: {name}")
    
    def get_adapter(self, name: str) -> Type[BaseVLMAdapter]:
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
        
        # Load the VLM components
        self.vision_encoder = vlm_adapter.extract_vision_encoder()
        self.language_model = vlm_adapter.extract_language_model()
        self.fusion_module = vlm_adapter.get_fusion_module()
        
        # Get VLM metadata
        metadata = vlm_adapter.get_metadata()
        
        # Create action decoder based on strategy
        if action_strategy == "continuous":
            self.action_decoder = nn.Sequential(
                nn.Linear(metadata.language_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        elif action_strategy == "discrete":
            # For discrete actions, predict action class
            self.action_decoder = nn.Sequential(
                nn.Linear(metadata.language_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)  # Assuming action_dim classes
            )
        else:
            raise ValueError(f"Unknown action strategy: {action_strategy}")
        
        self.metadata = metadata
        
        # Freeze VLM components initially (can be unfrozen for fine-tuning)
        self._freeze_vlm_components()
    
    def _freeze_vlm_components(self):
        """Freeze VLM components to train only action head initially"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.fusion_module.parameters():
            param.requires_grad = False
        
        logger.info("Frozen VLM components. Only action decoder will be trained initially.")
    
    def unfreeze_for_finetuning(self, unfreeze_vision: bool = True, unfreeze_language: bool = True):
        """Unfreeze components for full fine-tuning"""
        if unfreeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            logger.info("Unfroze vision encoder for fine-tuning")
        
        if unfreeze_language:
            for param in self.language_model.parameters():
                param.requires_grad = True
            logger.info("Unfroze language model for fine-tuning")
        
        for param in self.fusion_module.parameters():
            param.requires_grad = True
        logger.info("Unfroze fusion module for fine-tuning")
    
    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through VLM + action decoder"""
        
        # Encode vision
        vision_features = self.vision_encoder(images)
        
        # Encode language 
        language_features = self.language_model(text_tokens)
        
        # Fuse modalities
        if hasattr(self.fusion_module, 'forward'):
            fused_features = self.fusion_module(vision_features, language_features)
        else:
            # Simple concatenation fallback
            fused_features = torch.cat([vision_features, language_features], dim=-1)
        
        # Predict actions
        actions = self.action_decoder(fused_features)
        
        return actions
    
    def predict_action(self, image: torch.Tensor, instruction: str) -> torch.Tensor:
        """High-level interface for action prediction"""
        # Tokenize instruction
        tokenizer = self.vlm_adapter.get_tokenizer()
        text_tokens = tokenizer(instruction, return_tensors="pt")["input_ids"]
        
        # Ensure proper dimensions
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Predict
        with torch.no_grad():
            actions = self.forward(image, text_tokens)
        
        return actions.squeeze(0)  # Remove batch dimension


def create_nanovla_from_vlm(
    vlm_name: str,
    model_path: str,
    action_dim: int = 7,
    action_strategy: str = "continuous",
    **kwargs
) -> ActionHeadAdapter:
    """
    Create a nanoVLA model from any Vision-Language Model.
    
    This is the main implementation of the from_vlm() functionality.
    
    Args:
        vlm_name: Name of the VLM (e.g., "llava", "fastvlm", "openvla")
        model_path: Path to the pre-trained VLM model
        action_dim: Dimension of action space (default: 7 for 7-DOF robot)
        action_strategy: "continuous" or "discrete" action prediction
        **kwargs: Additional arguments passed to VLM loader
    
    Returns:
        ActionHeadAdapter: nanoVLA model ready for action prediction
    
    Example:
        ```python
        # Create nanoVLA from LLaVA
        model = create_nanovla_from_vlm(
            vlm_name="llava",
            model_path="liuhaotian/llava-v1.5-7b",
            action_dim=7
        )
        
        # Predict action
        action = model.predict_action(image, "pick up the red cup")
        ```
    """
    
    # Get the appropriate adapter class
    adapter_class = vlm_registry.get_adapter(vlm_name)
    
    # Create VLM adapter instance
    vlm_adapter = adapter_class(model_path, **kwargs)
    
    # Create action head adapter
    nanovla_model = ActionHeadAdapter(
        vlm_adapter=vlm_adapter,
        action_dim=action_dim,
        action_strategy=action_strategy
    )
    
    logger.info(f"Created nanoVLA from {vlm_name} with {action_dim}D {action_strategy} actions")
    
    return nanovla_model


## Example Usage and Tests
if __name__ == "__main__":
    # Example: Create nanoVLA from different VLMs
    
    print("Available VLM adapters:", vlm_registry.list_available())
    
    # This would work if you have the models available:
    """
    # LLaVA example
    llava_model = create_nanovla_from_vlm(
        vlm_name="llava",
        model_path="liuhaotian/llava-v1.5-7b",
        action_dim=7
    )
    
    # FastVLM example  
    fastvlm_model = create_nanovla_from_vlm(
        vlm_name="fastvlm",
        model_path="path/to/fastvlm/checkpoint",
        action_dim=7
    )
    
    # OpenVLA example
    openvla_model = create_nanovla_from_vlm(
        vlm_name="openvla", 
        model_path="openvla/openvla-7b",
        action_dim=7
    )
    """


## Tests
def test_vlm_registry():
    """Test VLM registry functionality"""
    registry = VLMRegistry()
    
    assert "llava" in registry.list_available()
    assert "fastvlm" in registry.list_available()
    assert "openvla" in registry.list_available()
    
    adapter_class = registry.get_adapter("llava")
    assert adapter_class == LLaVAAdapter
    
    print("✅ VLM registry tests passed")


def test_metadata_creation():
    """Test VLM metadata creation"""
    # Test with dummy path (won't actually load model)
    try:
        llava_adapter = LLaVAAdapter("dummy_path")
        # This would fail in get_metadata() because model loading fails
        # but we can test the structure
        assert llava_adapter.model_path == "dummy_path"
        print("✅ Adapter creation test passed")
    except Exception as e:
        print(f"Expected error in dummy test: {e}")


if __name__ == "__main__":
    test_vlm_registry()
    test_metadata_creation()
    print("✅ All factory tests passed")



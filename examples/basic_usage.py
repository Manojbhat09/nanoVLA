#!/usr/bin/env python3
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
Basic usage example for nanoVLA.

This demonstrates the key features of our professional VLA framework.
"""

import sys
from pathlib import Path
import numpy as np

# Add the source directory to Python path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)


def demonstrate_registry_system():
    """Demonstrate the advanced registry system"""
    print_header("Registry System Demo")
    
    try:
        from nanovla.registry import vlm_registry, CompatibilityLevel, VLMCapabilities
        
        print("🤖 Available VLMs in registry:")
        available_vlms = vlm_registry.list_available()
        for vlm in available_vlms:
            print(f"  • {vlm}")
        
        print("\n🔍 Registry information:")
        print(vlm_registry.get_info_table())
        
        # Demonstrate compatibility checking
        print("\n⚖️ Compatibility checking:")
        for vlm_name in ["nanovla", "llava", "blip2"]:
            try:
                compatibility = vlm_registry.check_compatibility(vlm_name, action_dim=7)
                print(f"  {vlm_name}: {compatibility.name}")
            except:
                print(f"  {vlm_name}: Not registered")
        
        print("✅ Registry system working!")
        
    except ImportError as e:
        print(f"❌ Registry demo failed: {e}")


def demonstrate_configuration():
    """Demonstrate configuration system"""
    print_header("Configuration System Demo")
    
    try:
        from nanovla.models.configuration_nanovla import nanoVLAConfig
        
        # Create default config
        config = nanoVLAConfig()
        print("📋 Default configuration:")
        print(f"  Action dimension: {config.action_dim}")
        print(f"  Hidden dimension: {config.hidden_dim}")
        print(f"  Vision model: {config.vision_model}")
        print(f"  Language model: {config.language_model}")
        print(f"  Fusion type: {config.fusion_type}")
        
        # Show model info
        print("\n📊 Model information:")
        model_info = config.get_model_info()
        for key, value in model_info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        print("✅ Configuration system working!")
        
    except ImportError as e:
        print(f"❌ Configuration demo failed: {e}")


def demonstrate_model_creation():
    """Demonstrate model creation with builder pattern"""
    print_header("Model Creation Demo")
    
    try:
        from nanovla.models.modeling_nanovla import nanoVLA, VLABuilder, VLAFactory
        from nanovla.models.configuration_nanovla import nanoVLAConfig
        
        print("🏗️ Method 1: Direct instantiation")
        config = nanoVLAConfig(action_dim=7, hidden_dim=512)
        model = nanoVLA(config)
        print(f"  Created model with {model.num_parameters():,} parameters")
        
        print("\n🏗️ Method 2: Builder pattern")
        try:
            model = (VLABuilder()
                    .with_vlm("nanovla")
                    .with_action_space(7, normalize=True)
                    .with_transfer_learning("fine_tune_all")
                    .with_fusion_strategy("cross_attention", num_heads=8)
                    .build())
            print(f"  Built model with {model.num_parameters():,} parameters")
        except Exception as e:
            print(f"  Builder pattern demo skipped: {e}")
        
        print("\n🏗️ Method 3: Factory methods")
        lightweight_model = VLAFactory.create_lightweight_model(action_dim=7)
        print(f"  Lightweight model: {lightweight_model.num_parameters():,} parameters")
        
        # Demonstrate model info
        print("\n📊 Model details:")
        model_info = model.get_model_info()
        print(f"  Device: {model_info['device']}")
        print(f"  Total parameters: {model_info['actual_parameters']['total']}")
        print(f"  Parameter size: {model_info['actual_parameters']['total_m']}")
        
        print("✅ Model creation working!")
        
    except ImportError as e:
        print(f"❌ Model creation demo failed: {e}")


def demonstrate_action_prediction():
    """Demonstrate action prediction"""
    print_header("Action Prediction Demo")
    
    try:
        from nanovla.models.modeling_nanovla import nanoVLA
        from nanovla.models.configuration_nanovla import nanoVLAConfig
        import torch
        
        # Create a simple model
        config = nanoVLAConfig(action_dim=7, vision_image_size=64)  # Smaller for demo
        model = nanoVLA(config)
        model.eval()
        
        print("🤖 Predicting robot action...")
        
        # Create dummy input data
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        instruction = "Pick up the red cup"
        
        # Predict action
        action = model.predict_action(dummy_image, instruction)
        
        print(f"  Image shape: {dummy_image.shape}")
        print(f"  Instruction: '{instruction}'")
        print(f"  Predicted action: {action}")
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        # Demonstrate forward pass
        print("\n🔄 Forward pass demo:")
        images = torch.rand(1, 3, 64, 64)
        text_tokens = torch.randint(0, 1000, (1, 32))
        
        with torch.no_grad():
            outputs = model.forward(images, text_tokens, return_dict=True)
        
        print(f"  Vision features shape: {outputs['vision_features'].shape}")
        print(f"  Language features shape: {outputs['language_features'].shape}")
        print(f"  Fused features shape: {outputs['fused_features'].shape}")
        print(f"  Actions shape: {outputs['actions'].shape}")
        
        print("✅ Action prediction working!")
        
    except ImportError as e:
        print(f"❌ Action prediction demo failed: {e}")
    except Exception as e:
        print(f"❌ Action prediction error: {e}")


def demonstrate_compatibility_checking():
    """Demonstrate VLM compatibility analysis"""
    print_header("VLM Compatibility Analysis")
    
    try:
        from nanovla.registry import (
            vlm_registry, 
            find_compatible_vlms, 
            recommend_vlm_for_task,
            validate_vlm_requirements
        )
        
        print("🔍 Finding compatible VLMs for 7-DOF robot:")
        compatible_vlms = find_compatible_vlms(action_dim=7)
        for vlm in compatible_vlms:
            print(f"  • {vlm}")
        
        print("\n🎯 VLM recommendations for robot manipulation:")
        recommendations = recommend_vlm_for_task(
            task_type="robot_manipulation",
            action_dim=7,
            performance_priority="balanced"
        )
        
        for i, vlm in enumerate(recommendations[:3], 1):
            print(f"  {i}. {vlm}")
        
        print("\n✅ Compatibility analysis working!")
        
    except ImportError as e:
        print(f"❌ Compatibility demo failed: {e}")


def demonstrate_import_system():
    """Demonstrate professional import system"""
    print_header("Import System Demo")
    
    try:
        from nanovla.utils.import_utils import (
            is_torch_available,
            is_transformers_available,
            is_vision_available,
            get_available_devices
        )
        
        print("🔧 Dependency checking:")
        print(f"  PyTorch available: {is_torch_available()}")
        print(f"  Transformers available: {is_transformers_available()}")
        print(f"  Vision dependencies available: {is_vision_available()}")
        
        print(f"\n💻 Available devices: {get_available_devices()}")
        
        # Demonstrate lazy loading
        print("\n📦 Lazy loading demonstration:")
        import nanovla
        print(f"  nanoVLA version: {nanovla.__version__}")
        print("  Modules are loaded on-demand for fast imports!")
        
        print("✅ Import system working!")
        
    except ImportError as e:
        print(f"❌ Import system demo failed: {e}")


def run_all_demos():
    """Run all demonstration functions"""
    print_header("nanoVLA Professional Framework Demo")
    print("🚀 Welcome to nanoVLA - Professional VLA Development")
    print("🎯 Demonstrating key features of our framework")
    
    # Run all demos
    demonstrate_import_system()
    demonstrate_registry_system()
    demonstrate_configuration()
    demonstrate_model_creation()
    demonstrate_action_prediction()
    demonstrate_compatibility_checking()
    
    print_header("Demo Complete!")
    print("🎉 All systems operational!")
    print("📚 Check README.md for more detailed usage")
    print("🤝 Ready for professional VLA development!")


def main():
    """Main function"""
    try:
        run_all_demos()
        return 0
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())



# Contributing to nanoVLA

Thank you for your interest in contributing to nanoVLA! This project aims to democratize Vision-Language-Action model development and we welcome contributions from the community.

## 🎯 **Mission**

nanoVLA strives to be the "nanoGPT of VLA models" - making robot learning accessible, educational, and extensible for everyone from students to researchers to industry practitioners.

## 🤝 **How to Contribute**

### **Types of Contributions Welcome**

1. **🐛 Bug Reports & Fixes**
   - Report issues with installation, usage, or documentation
   - Fix bugs in the core framework or examples

2. **📝 Documentation Improvements**
   - Improve clarity of existing documentation
   - Add examples and tutorials
   - Translate documentation

3. **🧪 VLM Support**
   - Add adapters for new Vision-Language Models
   - Improve existing VLM integrations
   - Test compatibility with different model versions

4. **🚀 Performance Optimizations**
   - Speed up inference or training
   - Reduce memory usage
   - Add quantization or deployment optimizations

5. **🤖 Real Robot Integration**
   - Test on actual robot hardware
   - Add ROS/simulation interfaces
   - Contribute robot-specific configurations

6. **🔬 Research Contributions**
   - Novel training strategies
   - Advanced action representations
   - Uncertainty quantification improvements

## 🚀 **Getting Started**

### **Development Setup**

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/nanovla.git
cd nanovla

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Run tests to verify setup
python test_installation.py
```

### **Development Workflow**

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines below
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run basic tests
   python test_installation.py
   
   # Run specific tests if available
   python -m pytest tests/ -v
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 **Code Style Guidelines**

### **Python Code Standards**

- **Type Annotations**: Use type hints throughout
- **Docstrings**: Use NumPy-style docstrings for functions and classes
- **Formatting**: Follow PEP 8 (use `black` for auto-formatting)
- **Imports**: Use absolute imports, organize with `isort`
- **Line Length**: Maximum 88 characters (black default)

### **Example Code Style**

```python
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image

class ActionDecoder(nn.Module):
    """
    Decode multimodal features to robot actions.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    action_dim : int
        Output action dimension
    hidden_dim : int, optional
        Hidden layer dimension, by default 512
        
    Examples
    --------
    >>> decoder = ActionDecoder(768, 7)
    >>> features = torch.randn(32, 768)
    >>> actions = decoder(features)
    >>> print(actions.shape)  # torch.Size([32, 7])
    """
    
    def __init__(
        self, 
        input_dim: int, 
        action_dim: int, 
        hidden_dim: int = 512
    ) -> None:
        super().__init__()
        self.action_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict actions from features."""
        return self.action_head(features)
```

### **Documentation Standards**

- **README**: Keep the main README concise but comprehensive
- **Code Comments**: Explain *why*, not *what*
- **Examples**: Include working code examples
- **Architecture**: Document design decisions and trade-offs

## 🧪 **Adding New VLM Support**

To add support for a new Vision-Language Model:

### **1. Create VLM Adapter**

```python
# In nanovla/src/nanovla/vlm_adapters/my_vlm.py

from ..registry import vlm_registry, VLMCapabilities, CompatibilityLevel
from ..models.modeling_utils import BaseVLM

@vlm_registry.register(
    "my_vlm",
    capabilities=VLMCapabilities(
        supports_vision=True,
        supports_language=True,
        compatibility_level=CompatibilityLevel.COMPATIBLE,
        transfer_learning_strategy="adapter_layers"
    )
)
class MyVLMAdapter(BaseVLM):
    """Adapter for MyVLM model."""
    
    def load_model(self, model_path: str):
        """Load the pre-trained model."""
        # Implementation here
        pass
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to features."""
        # Implementation here
        pass
    
    def encode_language(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode text tokens to features."""
        # Implementation here
        pass
```

### **2. Add Tests**

```python
# In tests/test_my_vlm.py

def test_my_vlm_integration():
    """Test that MyVLM can be loaded and used."""
    model = nanoVLA.from_vlm("my_vlm", action_dim=7)
    assert model is not None
    
    # Test basic functionality
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = "pick up cup"
    action = model.predict_action(dummy_image, dummy_text)
    assert action.shape == (1, 7)
```

### **3. Update Documentation**

- Add your VLM to the compatibility matrix in README.md
- Include usage examples
- Document any special requirements or limitations

## 🎯 **Contribution Guidelines**

### **Pull Request Guidelines**

1. **Clear Description**: Explain what your PR does and why
2. **Small, Focused Changes**: One feature/fix per PR
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update docs for user-facing changes
5. **Backwards Compatibility**: Don't break existing APIs without good reason

### **Commit Message Format**

Use clear, descriptive commit messages:

```
Category: Brief description

More detailed explanation if needed.

- List specific changes
- Include relevant issue numbers (#123)
```

**Categories**: Add, Fix, Update, Remove, Refactor, Test, Doc

### **Code Review Process**

1. **Automated Checks**: All tests must pass
2. **Peer Review**: At least one maintainer review required
3. **Discussion**: We may ask questions or suggest improvements
4. **Iteration**: You may need to make changes based on feedback

## 🤖 **Real Robot Testing**

If you have access to robot hardware:

### **Hardware Contributions**

1. **Test nanoVLA on your robot setup**
2. **Document hardware-specific configurations**
3. **Share performance results and videos**
4. **Contribute robot-specific adapters**

### **Simulation Testing**

Even without hardware, you can contribute:

1. **Test in simulation environments** (PyBullet, MuJoCo, etc.)
2. **Create new simulation scenarios**
3. **Benchmark performance on standardized tasks**

## 📊 **Research Contributions**

For research-oriented contributions:

### **Novel Algorithms**

1. **Uncertainty quantification** for safe robot control
2. **Efficient fusion architectures** for real-time inference
3. **Transfer learning strategies** for cross-embodiment learning
4. **Action representations** for better generalization

### **Empirical Studies**

1. **VLM comparison studies** across different robot tasks
2. **Scaling laws** for VLA model performance
3. **Real robot evaluation** on standardized benchmarks

### **Datasets & Benchmarks**

1. **New robot datasets** in RLDS format
2. **Evaluation protocols** for VLA models
3. **Sim-to-real transfer** studies

## 🐛 **Bug Reports**

When reporting bugs, please include:

### **Environment Information**
- Python version
- PyTorch version
- Operating system
- GPU/hardware details

### **Reproduction Steps**
```python
# Minimal code to reproduce the issue
import nanovla

model = nanovla.from_vlm("llava", action_dim=7)
# ... steps that cause the bug
```

### **Expected vs Actual Behavior**
- What you expected to happen
- What actually happened
- Error messages or stack traces

## 📞 **Communication**

### **Where to Ask Questions**

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and brainstorming
- **Pull Request Comments**: Code-specific discussions

### **Community Guidelines**

- **Be Respectful**: We welcome contributors from all backgrounds
- **Be Patient**: Reviews take time, especially for complex changes
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Inclusive**: Help create a welcoming environment for everyone

## 🎉 **Recognition**

Contributors will be recognized in:

- **README.md**: Contributors section
- **Release Notes**: Highlighting major contributions
- **Academic Papers**: Co-authorship for significant research contributions

## 📚 **Resources**

### **Learning Resources**
- [Vision-Language Models Overview](temp/docs/VLM_compatibility_analysis.md)
- [nanoVLA Architecture](temp/docs/nanoVLA_architecture.md)
- [Transfer Learning Guide](temp/docs/FROM_VLM_IMPLEMENTATION.md)

### **Development Tools**
- **Code Formatting**: `black`, `isort`
- **Type Checking**: `mypy`
- **Testing**: `pytest`
- **Documentation**: `sphinx` (coming soon)

---

Thank you for contributing to nanoVLA! Together, we're making robot learning accessible to everyone. 🤖✨

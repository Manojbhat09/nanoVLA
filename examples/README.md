# nanoVLA Examples

This directory contains examples and demonstrations of nanoVLA functionality.

## 📋 **Available Examples**

### **Core Functionality**
- **`test_installation.py`** - Verify nanoVLA installation and basic functionality
- **`nanovla_main_integration.py`** - Main integration examples and usage patterns
- **`nanovla_vlm_factory.py`** - VLM factory system demonstration
- **`nanovla_rl_training.py`** - RL training integration examples

### **From temp/examples/** (moved during cleanup)
- **`demo_from_vlm.py`** - Comprehensive demo of from_vlm() functionality
- **`vlm_integration_examples.py`** - Multiple VLM integration examples
- **`test_from_vlm_minimal.py`** - Minimal testing for from_vlm()
- **`create_showcase_images.py`** - Generate visualization demos

## 🚀 **Quick Start**

```bash
# Test basic installation
python examples/test_installation.py

# Try VLM integration
python examples/demo_from_vlm.py

# Test specific functionality
python examples/test_from_vlm_minimal.py
```

## 📚 **Example Categories**

### **Basic Usage**
```python
# Simple model creation
import sys
sys.path.insert(0, 'nanovla/src')
import nanovla

model = nanovla.nanoVLA.from_vlm("llava", action_dim=7)
```

### **Advanced Configuration**
```python
# Custom VLA builder
model = (nanovla.VLABuilder()
        .with_vlm("llava", model_size="7b")
        .with_action_space(7, bounds=[[-1, 1]] * 7)
        .with_transfer_learning("adapter_layers")
        .build())
```

### **RL Training**
```python
# RL training setup
from examples.nanovla_rl_training import nanoVLA_RL_Trainer
trainer = nanoVLA_RL_Trainer(config)
trainer.train()
```

See individual files for detailed examples and usage patterns.

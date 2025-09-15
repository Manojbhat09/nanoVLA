# 🛠️ nanoVLA Installation Guide

## 📋 Quick Start

### Option 1: Direct Usage (Recommended for Development)

```bash
# Clone the repository
git clone <repository-url>
cd nanoVLA

# Add the package to your Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/nanovla/src"

# Test the installation
python test_installation.py
```

### Option 2: Virtual Environment Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Test the installation  
python test_installation.py
```

### Option 3: System Installation (if permitted)

```bash
# Install with system packages override (use with caution)
pip install -e . --break-system-packages

# Or use pipx for isolated installation
pipx install .
```

## 🚀 Usage Examples

### Basic Import and Model Creation

```python
import sys
sys.path.insert(0, 'nanovla/src')  # If using Option 1
import nanovla

# Create a nanoVLA model from any VLM
model = nanovla.nanoVLA.from_vlm(
    vlm_name="llava",
    vlm_path="path/to/llava/checkpoint",
    action_dim=7,  # Robot action dimensions
    device="cuda"
)

# Use the model for vision-language-action prediction
actions = model.predict_actions(images, instructions)
```

### Available VLMs

```python
# Check supported VLMs
from nanovla.vlm_factory import vlm_registry
print("Available VLMs:", vlm_registry.list_available())
# Output: ['llava', 'fastvlm', 'openvla', 'llava-1.5', 'llava-1.6']
```

### RL Training Integration

```python
from nanovla.training import nanoVLA_RL_Trainer, create_rl_training_config

# Create RL training configuration
config = create_rl_training_config(
    sft_model_path="path/to/sft/model",
    dataset_name="libero_10",
    experiment_name="my_rl_experiment"
)

# Initialize trainer
trainer = nanoVLA_RL_Trainer(config)

# Start RL training
trainer.train()
```

## 🔧 Dependencies

### Core Requirements
- Python >= 3.9
- PyTorch >= 2.0.0
- Transformers >= 4.40.0
- Pillow >= 8.0.0
- NumPy >= 1.20.0

### Optional Dependencies

Install additional features as needed:

```bash
# Vision processing
pip install torchvision opencv-python albumentations

# Robotics simulation
pip install gym pybullet mujoco

# Training & Logging  
pip install wandb tensorboard accelerate datasets

# RL Training
pip install verl ray
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Make sure to add the package to your Python path:
   ```python
   import sys
   sys.path.insert(0, 'nanovla/src')
   ```

2. **Terminal Hanging in Cursor**: If experiencing terminal issues:
   ```bash
   unset VSCODE_SHELL_INTEGRATION
   export TERM=xterm-256color
   ```

3. **Logging Errors**: If you see logger-related errors, ensure you're using the latest version with fixed imports.

4. **Externally Managed Environment**: Use virtual environments or `--break-system-packages` flag.

### Testing Installation

Run the comprehensive test suite:

```bash
python test_installation.py
```

Expected output:
```
🧪 Testing nanoVLA Installation
========================================
✅ Passed: 4/4 tests
🎉 All tests passed! nanoVLA is ready to use.
```

## 📚 Next Steps

- Check out the [Examples](vlm_integration_examples.py) for usage patterns
- Read the [Architecture Guide](nanoVLA_architecture.md) to understand the framework
- Explore [VLM Compatibility](VLM_compatibility_analysis.md) for supported models
- See [RL Training Guide](nanovla/src/nanovla/training/rl_training.py) for advanced training

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test_installation.py`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

**Happy coding with nanoVLA! 🎉**

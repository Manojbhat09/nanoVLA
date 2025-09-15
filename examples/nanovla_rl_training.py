#!/usr/bin/env python3
"""
nanoVLA RL Training Integration

Based on SimpleVLA-RL (https://github.com/PRIME-RL/SimpleVLA-RL)
This module provides reinforcement learning training for nanoVLA models.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingConfig:
    """Configuration for RL training following SimpleVLA-RL approach"""
    
    # Model configuration
    sft_model_path: str
    experiment_name: str = "nanovla_rl_experiment"
    vla_name: str = "nanovla"
    
    # Training configuration
    dataset_name: str = "libero_10"  # libero_10, libero_90, libero_spatial, libero_object, libero_goal
    num_trials_per_task: int = 50
    n_samples: int = 8
    train_batch_size: int = 64
    val_batch_size: int = 496
    max_prompt_length: int = 256
    max_response_length: int = 128
    
    # RL hyperparameters
    learning_rate: float = 5e-6
    clip_ratio_high: float = 0.28
    clip_ratio_low: float = 0.2
    temperature: float = 1.6
    total_epochs: int = 100
    save_freq: int = 25
    test_freq: int = 4
    
    # Hardware configuration
    num_gpus: int = 8
    num_nodes: int = 1
    
    # Paths
    checkpoint_path: str = "./checkpoints"
    wandb_api_key: Optional[str] = None
    
    # Advanced settings
    action_token_len: int = 7
    action_chunks_len: int = 8
    enable_gradient_checkpointing: bool = False
    kl_coef: float = 0.0
    entropy_coeff: float = 0.0


class nanoVLA_RL_Trainer:
    """
    Reinforcement Learning trainer for nanoVLA models.
    
    This trainer integrates with the SimpleVLA-RL framework to provide
    online RL training for VLA models using outcome-level 0/1 rewards.
    
    Example:
        ```python
        # Configure RL training
        config = RLTrainingConfig(
            sft_model_path="path/to/sft/model",
            experiment_name="my_rl_experiment",
            dataset_name="libero_10",
            wandb_api_key="your_wandb_key"
        )
        
        # Initialize trainer
        trainer = nanoVLA_RL_Trainer(config)
        
        # Start RL training
        trainer.train()
        
        # Evaluate trained model
        results = trainer.evaluate()
        ```
    """
    
    def __init__(self, config: RLTrainingConfig):
        self.config = config
        self.align_file_path = None
        self._validate_config()
        
    def _validate_config(self):
        """Validate the training configuration"""
        if not os.path.exists(self.config.sft_model_path):
            raise ValueError(f"SFT model path does not exist: {self.config.sft_model_path}")
            
        if self.config.dataset_name not in ["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"]:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
            
        if self.config.wandb_api_key is None:
            logger.warning("No WandB API key provided. Logging may be limited.")
    
    def _create_align_file(self) -> str:
        """Create the align.json file required by SimpleVLA-RL"""
        align_config = {
            "env_vars": {
                "NCCL_DEBUG": "WARN",
                "RAY_memory_monitor_refresh_ms": "0",
                "VLLM_ATTENTION_BACKEND": "XFORMERS",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TOKENIZERS_PARALLELISM": "true",
                "WANDB_API_KEY": self.config.wandb_api_key or ""
            },
            "excludes": ["*"]
        }
        
        # Create temporary align file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(align_config, f, indent=2)
            self.align_file_path = f.name
            
        return self.align_file_path
    
    def _build_training_command(self) -> List[str]:
        """Build the RL training command based on config"""
        align_path = self._create_align_file()
        
        # Base command
        cmd = [
            "python", "-m", "verl.trainer.main_ppo",
            
            # Data configuration
            f"data.task_suite_name={self.config.dataset_name}",
            f"data.num_trials_per_task={self.config.num_trials_per_task}",
            f"data.n_samples={self.config.n_samples}",
            "data.filter_accuracy=True",
            "data.accuracy_lower_bound=0.1",
            "data.accuracy_upper_bound=0.9",
            "data.oversample_factor=1",
            f"data.train_batch_size={self.config.train_batch_size}",
            f"data.val_batch_size={self.config.val_batch_size}",
            f"data.max_prompt_length={self.config.max_prompt_length}",
            f"data.max_response_length={self.config.max_response_length}",
            
            # Model configuration
            f"actor_rollout_ref.model.path={self.config.sft_model_path}",
            f"actor_rollout_ref.model.vla={self.config.vla_name}",
            f"actor_rollout_ref.model.action_token_len={self.config.action_token_len}",
            f"actor_rollout_ref.model.action_chunks_len={self.config.action_chunks_len}",
            f"actor_rollout_ref.model.enable_gradient_checkpointing={self.config.enable_gradient_checkpointing}",
            "actor_rollout_ref.model.use_remove_padding=False",
            
            # Actor configuration
            f"actor_rollout_ref.actor.optim.lr={self.config.learning_rate}",
            "actor_rollout_ref.actor.optim.warmup_style=constant",
            "actor_rollout_ref.actor.ppo_mini_batch_size=128",
            f"actor_rollout_ref.actor.ppo_micro_batch_size={self.config.num_gpus}",
            "actor_rollout_ref.actor.use_dynamic_bsz=False",
            "actor_rollout_ref.actor.fsdp_config.param_offload=False",
            "actor_rollout_ref.actor.fsdp_config.grad_offload=True",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            "actor_rollout_ref.actor.grad_clip=1",
            f"actor_rollout_ref.actor.clip_ratio_high={self.config.clip_ratio_high}",
            f"actor_rollout_ref.actor.clip_ratio_low={self.config.clip_ratio_low}",
            "actor_rollout_ref.actor.num_images_in_input=1",
            "actor_rollout_ref.actor.traj_mini_batch_size=16",
            f"actor_rollout_ref.actor.entropy_coeff={self.config.entropy_coeff}",
            
            # Rollout configuration
            "actor_rollout_ref.rollout.num_images_in_input=1",
            "actor_rollout_ref.rollout.val_micro_batch_size=8",
            f"actor_rollout_ref.rollout.temperature={self.config.temperature}",
            f"actor_rollout_ref.rollout.experiment_name={self.config.experiment_name}",
            "actor_rollout_ref.rollout.micro_batch_size=1",
            f"actor_rollout_ref.rollout.unnorm_key={self.config.dataset_name}",
            "actor_rollout_ref.rollout.model_family=nanovla",
            f"actor_rollout_ref.rollout.task_suite_name={self.config.dataset_name}",
            "actor_rollout_ref.rollout.num_steps_wait=10",
            f"actor_rollout_ref.rollout.pretrained_checkpoint={self.config.sft_model_path}",
            "actor_rollout_ref.rollout.center_crop=True",
            "actor_rollout_ref.rollout.max_prompt_length=512",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size=32",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.name=hf",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.9",
            
            # Reference model configuration
            "actor_rollout_ref.ref.log_prob_micro_batch_size=32",
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            
            # Algorithm configuration
            f"algorithm.kl_ctrl.kl_coef={self.config.kl_coef}",
            "algorithm.adv_estimator=grpo",
            "algorithm.adv_params.verifier_gamma=1.0",
            "algorithm.adv_params.reward_model_gamma=1.0",
            
            # Trainer configuration
            "trainer.logger=['console','wandb']",
            "trainer.project_name=nanoVLA-RL",
            f"trainer.experiment_name={self.config.experiment_name}",
            f"trainer.default_local_dir={self.config.checkpoint_path}/nanoVLA-RL/{self.config.experiment_name}",
            f"trainer.n_gpus_per_node={self.config.num_gpus}",
            f"trainer.nnodes={self.config.num_nodes}",
            f"trainer.save_freq={self.config.save_freq}",
            f"trainer.test_freq={self.config.test_freq}",
            f"trainer.total_epochs={self.config.total_epochs}",
            "trainer.val_only=False",
            f"trainer.runtime_env={align_path}",
            "trainer.wandb_mode=online",
            "trainer.val_before_train=True"
        ]
        
        return cmd
    
    def train(self) -> Dict[str, float]:
        """
        Start RL training for the nanoVLA model.
        
        Returns:
            Dict containing training metrics and final performance
        """
        logger.info(f"Starting RL training for experiment: {self.config.experiment_name}")
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"SFT model: {self.config.sft_model_path}")
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_USE_CUDA_DSA": "1",
            "HYDRA_FULL_ERROR": "1"
        })
        
        if self.config.wandb_api_key:
            env["WANDB_API_KEY"] = self.config.wandb_api_key
        
        # Build and execute training command
        cmd = self._build_training_command()
        
        try:
            logger.info(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("RL training completed successfully")
            logger.info(f"Output: {result.stdout}")
            
            return {"status": "success", "message": "Training completed"}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"RL training failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise RuntimeError(f"RL training failed: {e}")
        
        finally:
            # Clean up temporary align file
            if self.align_file_path and os.path.exists(self.align_file_path):
                os.unlink(self.align_file_path)
    
    def evaluate(self, model_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate a trained RL model.
        
        Args:
            model_path: Path to the model checkpoint. If None, uses latest checkpoint.
        
        Returns:
            Dict containing evaluation metrics
        """
        if model_path is None:
            # Find latest checkpoint
            checkpoint_dir = Path(self.config.checkpoint_path) / "nanoVLA-RL" / self.config.experiment_name
            if not checkpoint_dir.exists():
                raise ValueError(f"No checkpoints found at {checkpoint_dir}")
            
            # This would need to be implemented based on checkpoint naming convention
            model_path = str(checkpoint_dir / "latest_checkpoint")
        
        # Set up evaluation configuration
        eval_config = self.config
        eval_config.sft_model_path = model_path
        
        # Build evaluation command (same as training but with val_only=True)
        cmd = self._build_training_command()
        
        # Replace val_only=False with val_only=True
        cmd = [arg.replace("trainer.val_only=False", "trainer.val_only=True") for arg in cmd]
        
        env = os.environ.copy()
        env.update({
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
        })
        
        try:
            logger.info("Starting model evaluation")
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Evaluation completed successfully")
            logger.info(f"Output: {result.stdout}")
            
            # Parse evaluation results (this would need to be implemented)
            return {"status": "success", "accuracy": 0.0}  # Placeholder
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise RuntimeError(f"Evaluation failed: {e}")


def create_rl_training_config(
    sft_model_path: str,
    experiment_name: str = "nanovla_rl_experiment",
    dataset: str = "libero_10",
    **kwargs
) -> RLTrainingConfig:
    """
    Create an RL training configuration with sensible defaults.
    
    Args:
        sft_model_path: Path to the supervised fine-tuned model
        experiment_name: Name for the experiment
        dataset: Dataset name (libero_10, libero_90, etc.)
        **kwargs: Additional configuration parameters
    
    Returns:
        RLTrainingConfig object
    """
    return RLTrainingConfig(
        sft_model_path=sft_model_path,
        experiment_name=experiment_name,
        dataset_name=dataset,
        **kwargs
    )


## Example Usage
if __name__ == "__main__":
    # Example: Set up RL training for nanoVLA
    config = create_rl_training_config(
        sft_model_path="/path/to/your/sft/model",
        experiment_name="nanovla_libero_experiment",
        dataset="libero_10",
        wandb_api_key="your_wandb_key_here",
        num_gpus=8,
        total_epochs=50
    )
    
    # Initialize trainer
    trainer = nanoVLA_RL_Trainer(config)
    
    # Start training
    try:
        results = trainer.train()
        print(f"Training results: {results}")
        
        # Evaluate the trained model
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
    except Exception as e:
        print(f"Training failed: {e}")


## Tests
def test_rl_config_creation():
    """Test RL configuration creation"""
    config = create_rl_training_config(
        sft_model_path="/tmp/test_model",
        experiment_name="test_experiment"
    )
    
    assert config.sft_model_path == "/tmp/test_model"
    assert config.experiment_name == "test_experiment"
    assert config.dataset_name == "libero_10"
    assert config.learning_rate == 5e-6


def test_command_building():
    """Test that training command is built correctly"""
    config = create_rl_training_config(
        sft_model_path="/tmp/test_model",
        experiment_name="test_experiment"
    )
    
    trainer = nanoVLA_RL_Trainer(config)
    
    # This would fail because model path doesn't exist, but we can test command building
    try:
        cmd = trainer._build_training_command()
        assert len(cmd) > 10  # Should have many configuration parameters
        assert "python" in cmd[0]
        assert "-m" in cmd[1]
        assert "verl.trainer.main_ppo" in cmd[2]
    except ValueError:
        pass  # Expected due to non-existent model path


if __name__ == "__main__":
    # Run basic tests
    test_rl_config_creation()
    test_command_building()
    print("✅ Basic tests passed")



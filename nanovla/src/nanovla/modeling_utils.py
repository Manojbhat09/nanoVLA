# Copyright 2024 The nanoVLA Team. All rights reserved.
# Adapted from HuggingFace Transformers
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
Base classes for nanoVLA models, following transformers patterns.
"""

import copy
import functools
import inspect
import os
import time
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .configuration_utils import PretrainedConfig, VLAConfig
from .utils import logging


logger = logging.get_logger(__name__)


@runtime_checkable
class VLMProtocol(Protocol):
    """Protocol defining the interface that all VLMs must implement"""
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual input to feature representations"""
        ...
    
    def encode_language(self, text_tokens: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode language input to feature representations"""
        ...
    
    @property
    def vision_feature_dim(self) -> int:
        """Dimension of vision features"""
        ...
    
    @property
    def language_feature_dim(self) -> int:
        """Dimension of language features"""
        ...
    
    @property
    def supports_batch_processing(self) -> bool:
        """Whether the model supports batch processing"""
        ...


class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None

    def add_memory_hooks(self):
        """
        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

        Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero
        with `model.reset_memory_hooks_state()`.
        """
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        return self

    def reset_memory_hooks_state(self):
        """
        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).
        """
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

        return self

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return next(self.parameters()).dtype

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get the number of (optionally, trainable or non-embeddings) parameters in the model.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

        if is_loaded_in_4bit:
            if hasattr(self, "hf_quantizer"):
                total_numel = self.hf_quantizer.get_model_size_in_bytes(self) / 4
            else:
                # deprecated in v4.39.0
                warnings.warn(
                    "Detecting 4-bit quantized model without a `hf_quantizer` attribute. This is deprecated and will be removed in transformers v4.40. "
                    "Please make sure to pass a `quantization_config` when loading the model with `from_pretrained`.",
                    FutureWarning,
                )
                for param in total_parameters:
                    if param.requires_grad or not only_trainable:
                        # Each 4-bit weight has 0.5 bytes, and potentially additionnal memory overhead
                        if hasattr(param, "quant_state"):
                            total_numel.append(param.numel() * 0.5)
                            # Some 4-bit weights have an overhead, we need to take it into account
                            if hasattr(param.quant_state, "nested") and param.quant_state.nested:
                                total_numel.append(param.quant_state.absmax.numel())
                        else:
                            total_numel.append(param.numel())
                total_numel = sum(total_numel)
        else:
            for param in total_parameters:
                if param.requires_grad or not only_trainable:
                    total_numel.append(param.numel())
            total_numel = sum(total_numel)
        return int(total_numel)

    def estimate_tokens_per_second(
        self,
        input_ids: torch.Tensor,
        n_steps: int = 256,
        batch_sizes: Optional[List[int]] = None,
        sequence_lengths: Optional[List[int]] = None,
    ) -> float:
        """
        Measures the token rate (tokens per second) of the model on the given `input_ids`.

        Args:
            input_ids (`torch.Tensor`):
                The input ids to use for the benchmark. The tensor should be of shape `(batch_size, sequence_length)`.
            n_steps (`int`, *optional*, defaults to 256):
                The number of steps to run the benchmark for.
            batch_sizes (`List[int]`, *optional*):
                The batch sizes to test. If not provided, will use the batch size of the `input_ids`.
            sequence_lengths (`List[int]`, *optional*):
                The sequence lengths to test. If not provided, will use the sequence length of the `input_ids`.

        Returns:
            `float`: The number of tokens per second.
        """
        if batch_sizes is None:
            batch_sizes = [input_ids.shape[0]]
        if sequence_lengths is None:
            sequence_lengths = [input_ids.shape[1]]

        with torch.no_grad():
            for batch_size in batch_sizes:
                for sequence_length in sequence_lengths:
                    if batch_size * sequence_length > input_ids.numel():
                        logger.warning(
                            f"The product of batch_size ({batch_size}) and sequence_length ({sequence_length}) should not exceed the total number of tokens in input_ids ({input_ids.numel()}). Skipping this configuration."
                        )
                        continue

                    _input_ids = input_ids[:batch_size, :sequence_length]
                    # Warmup
                    for _ in range(10):
                        _ = self(_input_ids)

                    torch.cuda.synchronize()

                    start_time = time.perf_counter()

                    for _ in range(n_steps):
                        _ = self(_input_ids)

                    torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    return (n_steps * batch_size * sequence_length) / (end_time - start_time)


class PretrainedVLMMixin:
    """
    A mixin class for loading and saving VLM models using HuggingFace patterns.
    """

    config_class = None
    base_model_prefix = ""
    _no_split_modules = []
    _skip_keys_device_placement = []
    _keep_in_fp32_modules = []

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        """
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
        deactivated). To train the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in XXX* means that the layer XXX is not used by this configuration of
        the model, so those weights are discarded.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            _fast_init(`bool`, *optional*, defaults to `True`):
                Whether or not to disable fast initialization.

                <Tip warning={true}>

                One should only disable `_fast_init` to ensure backwards compatibility with `transformers.__version__ <
                4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
                [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

                </Tip>

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
        ```

        * `low_cpu_mem_usage` algorithm:

        This is an experimental function that loads the model using ~1x model size CPU memory

        Here is how it works:

        1. save which state_dict keys we have
        2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
        3. after the model has been instantiated switch to the meta device all the params/buffers that
        are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

        """
        # This is a simplified implementation - in a real scenario, you'd want to implement
        # the full loading logic following transformers patterns
        logger.info(f"Loading pretrained model from {pretrained_model_name_or_path}")
        
        # For now, just create a new instance with default config
        # In a real implementation, you'd load the actual weights and config
        if config is None:
            if hasattr(cls, 'config_class') and cls.config_class is not None:
                config = cls.config_class()
            else:
                config = VLAConfig()
        
        model = cls(config, *model_args, **kwargs)
        model.eval()
        
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to `"5GB"` so that users can easily load models on free-tier Google Colab instances
                without any CPU OOM issues.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, the
                weights are saved in PEFT format.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        logger.info(f"Saving model to {save_directory}")
        
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save from the main process
        if is_main_process:
            # Save the config
            if hasattr(self, 'config'):
                self.config.save_pretrained(save_directory)
            
            # Save the model weights (simplified)
            model_to_save = self
            if state_dict is None:
                state_dict = model_to_save.state_dict()

            # Save model weights
            weights_name = "pytorch_model.bin"
            if safe_serialization:
                weights_name = "model.safetensors"
                
            if variant is not None:
                weights_name = weights_name.replace("pytorch_model", f"pytorch_model.{variant}")
                weights_name = weights_name.replace("model.safetensors", f"model.{variant}.safetensors")

            weights_path = os.path.join(save_directory, weights_name)
            
            if safe_serialization:
                try:
                    from safetensors.torch import save_file
                    save_file(state_dict, weights_path)
                except ImportError:
                    logger.warning("safetensors not available, falling back to pickle")
                    save_function(state_dict, weights_path)
            else:
                save_function(state_dict, weights_path)

            logger.info(f"Model weights saved in {weights_path}")

        if push_to_hub:
            # Implementation for pushing to hub would go here
            logger.info("Push to hub functionality not implemented yet")


class CompatibilityMixin:
    """Mixin for VLM compatibility checking and adaptation"""
    
    def check_action_compatibility(self, action_dim: int):
        """Check if this VLM can be adapted for given action dimension"""
        from .registry import vlm_registry
        return vlm_registry.check_compatibility(self.__class__.__name__, action_dim)
    
    def suggest_transfer_strategy(self, target_task: str):
        """Suggest optimal transfer learning strategy"""
        from .registry import vlm_registry, TransferLearningStrategy
        
        capabilities = vlm_registry.get_capabilities(self.__class__.__name__)
        if capabilities is None:
            return TransferLearningStrategy.FINE_TUNE_ALL
        
        if capabilities.compatibility_level.name == "NATIVE":
            return TransferLearningStrategy.FINE_TUNE_ALL
        elif capabilities.compatibility_level.name == "COMPATIBLE":
            return TransferLearningStrategy.ADAPTER_LAYERS
        else:
            return TransferLearningStrategy.FREEZE_BACKBONE


class BaseVLM(nn.Module, ModuleUtilsMixin, ABC):
    """Abstract base class for all VLMs with common functionality"""
    
    config_class = VLAConfig
    base_model_prefix = "vlm"
    _no_split_modules = []
    _skip_keys_device_placement = []
    _keep_in_fp32_modules = []
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        self._vision_feature_dim: Optional[int] = None
        self._language_feature_dim: Optional[int] = None
    
    @abstractmethod
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual input"""
        pass
    
    @abstractmethod 
    def encode_language(self, text_tokens: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode language input"""
        pass
    
    @property
    @abstractmethod
    def vision_feature_dim(self) -> int:
        pass
    
    @property
    @abstractmethod
    def language_feature_dim(self) -> int:
        pass
    
    @property
    def supports_batch_processing(self) -> bool:
        return True
    
    def freeze_vision_encoder(self):
        """Utility to freeze vision encoder parameters"""
        for name, param in self.named_parameters():
            if 'vision' in name.lower() or 'visual' in name.lower():
                param.requires_grad = False
    
    def freeze_language_encoder(self):
        """Utility to freeze language encoder parameters"""  
        for name, param in self.named_parameters():
            if 'language' in name.lower() or 'text' in name.lower():
                param.requires_grad = False
    
    def get_input_embeddings(self):
        """Get input embeddings if available"""
        if hasattr(self, 'language_model') and hasattr(self.language_model, 'get_input_embeddings'):
            return self.language_model.get_input_embeddings()
        return None
    
    def set_input_embeddings(self, new_embeddings):
        """Set input embeddings if available"""
        if hasattr(self, 'language_model') and hasattr(self.language_model, 'set_input_embeddings'):
            self.language_model.set_input_embeddings(new_embeddings)
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self.get_input_embeddings()
        if model_embeds is None:
            return None

        if new_num_tokens is None:
            return model_embeds

        new_num_tokens = int(new_num_tokens)
        old_num_tokens, old_embedding_dim = model_embeds.weight.size()

        if old_num_tokens == new_num_tokens:
            return model_embeds

        if not isinstance(model_embeds, nn.Embedding):
            raise RuntimeError(
                f"Not supported embedding type for resizing: {type(model_embeds)}. Only `torch.nn.Embedding` is"
                " supported."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            model_embeds.weight.device, dtype=model_embeds.weight.dtype
        )

        # Initialize new embeddings
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = model_embeds.weight.data[:n, :]

        # Set new embeddings
        self.set_input_embeddings(new_embeddings)

        # Update config
        if hasattr(self.config, "vocab_size"):
            self.config.vocab_size = new_num_tokens

        # Tie weights if necessary
        if hasattr(self, "tie_weights"):
            self.tie_weights()

        return self.get_input_embeddings()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare inputs for generation. Can be overridden by subclasses.
        """
        return {"input_ids": input_ids}

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Initialize with normal distribution
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            logger.warning(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

        if getattr(self, "_hf_peft_config_loaded", False):
            # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
            # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
            # the gradients to make sure the gradient flows.
            self.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` method
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                logger.warning(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(partial(self._set_gradient_checkpointing, value=False))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()

    @property
    def supports_gradient_checkpointing(self):
        """
        Whether gradient checkpointing is supported by this model.
        """
        return hasattr(self, "_set_gradient_checkpointing")


# Main exports
__all__ = [
    "VLMProtocol",
    "ModuleUtilsMixin", 
    "PretrainedVLMMixin",
    "CompatibilityMixin",
    "BaseVLM",
]

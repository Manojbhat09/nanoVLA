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
Configuration classes for nanoVLA models.
"""

import copy
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Union

from .utils import logging


logger = logging.get_logger(__name__)


class PretrainedConfig:
    """
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    Note:
        A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
        initialize a model does **not** load the model weights. It only affects the model's configuration.

    Class attributes (overridden by derived classes):
        - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, and used to
          recreate the correct object in [`~PretrainedConfig.from_pretrained`].
        - **is_composition** (`bool`) -- Whether the config class is composed of multiple sub-configs. In this case
          the config has to be initialized from two or more configs of type [`~PretrainedConfig`] like:
          [`~VisionTextDualEncoderConfig`].

    Args:
        name_or_path (`str`, *optional*, defaults to `""`):
            Store the string that was passed to [`PreTrainedModel.from_pretrained`] or
            [`TFPreTrainedModel.from_pretrained`] as `pretrained_model_name_or_path` if the configuration was created
            with such a method.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return all hidden-states.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not the model should returns all attentions.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a [`~utils.ModelOutput`] instead of a plain tuple.
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        cross_attention_hidden_size** (`int`, *optional*):
            The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder
            model and the cross-attention hidden dimension differs from the hidden dimension.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the [`EncoderDecoderModel`] class, which consists of all models
            in `AUTO_MODELS_FOR_CAUSAL_LM`.
        tie_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (`Dict[int, List[int]]`, *optional*, defaults to `{}`):
            Pruned heads of the model. The keys are the selected layer indices and the values are the list of heads to
            prune in said layer.

            For example `{1: [0, 2], 2: [2, 3]}` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (`int`, *optional*, defaults to `0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that
            the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` <
            sequence_length embeddings at a time. For more information on feed forward chunking, see [How does feed
            forward chunking work?](../glossary.html#feed-forward-chunking).

    Parameters for sequence generation:
        - **max_length** (`int`, *optional*, defaults to 20) -- Maximum length that will be used by default in the
          `generate` method of the model.
        - **min_length** (`int`, *optional*, defaults to 10) -- Minimum length that will be used by default in the
          `generate` method of the model.
        - **do_sample** (`bool`, *optional*, defaults to `False`) -- Flag that will be used by default in the
          `generate` method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
        - **early_stopping** (`bool`, *optional*, defaults to `False`) -- Flag that will be used by default in the
          `generate` method of the model. Whether to stop the beam search when at least `num_beams` sentences are
          finished per batch or not.
        - **num_beams** (`int`, *optional*, defaults to 1) -- Number of beams for beam search that will be used by
          default in the `generate` method of the model. 1 means no beam search.
        - **num_beam_groups** (`int`, *optional*, defaults to 1) -- Number of groups to divide `num_beams` into in
          order to ensure diversity among different groups of beams that will be used by default in the `generate`
          method of the model. 1 means no group beam search.
        - **diversity_penalty** (`float`, *optional*, defaults to 0.0) -- Value to control diversity for group beam
          search. that will be used by default in the `generate` method of the model. 0 means no diversity penalty.
          The higher the penalty, the more diverse are the outputs.
        - **temperature** (`float`, *optional*, defaults to 1) -- The value used to module the next token
          probabilities that will be used by default in the `generate` method of the model. Must be strictly positive.
        - **top_k** (`int`, *optional*, defaults to 50) -- Number of highest probability vocabulary tokens to keep for
          top-k-filtering that will be used by default in the `generate` method of the model.
        - **top_p** (`float`, *optional*, defaults to 1) -- Value that will be used by default in the `generate`
          method of the model for `top_p`. If set to float < 1, only the most probable tokens with probabilities that
          add up to `top_p` or higher are kept for generation.
        - **typical_p** (`float`, *optional*, defaults to 1) -- Local typicality measures how similar the conditional
          probability of predicting a target token next is to the expected conditional probability of predicting a
          random token next, given the partial text already generated. If set to float < 1, the smallest set of the
          most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation.
          See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
        - **repetition_penalty** (`float`, *optional*, defaults to 1) -- Parameter for repetition penalty that will be
          used by default in the `generate` method of the model. 1.0 means no penalty.
        - **length_penalty** (`float`, *optional*, defaults to 1) -- Exponential penalty to the length that is used
          with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to
          divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative),
          `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.
        - **no_repeat_ngram_size** (`int`, *optional*, defaults to 0) -- Value that will be used by default in the
          `generate` method of the model for `no_repeat_ngram_size`. If set to int > 0, all ngrams of that size can
          only occur once.
        - **encoder_no_repeat_ngram_size** (`int`, *optional*, defaults to 0) -- Value that will be used by default in
          the `generate` method of the model for `encoder_no_repeat_ngram_size`. If set to int > 0, all ngrams of that
          size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.
        - **bad_words_ids** (`List[int]`, *optional*) -- List of token ids that are not allowed to be generated that
          will be used by default in the `generate` method of the model. In order to get the tokens of the words that
          should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.
        - **num_return_sequences** (`int`, *optional*, defaults to 1) -- Number of independently computed returned
          sequences for each element in the batch that will be used by default in the `generate` method of the model.
        - **output_scores** (`bool`, *optional*, defaults to `False`) -- Whether the model should return the logits
          when used for generation.
        - **return_dict_in_generate** (`bool`, *optional*, defaults to `False`) -- Whether the model should return a
          [`~utils.ModelOutput`] instead of a `torch.LongTensor`.
        - **forced_bos_token_id** (`int`, *optional*) -- The id of the token to force as the first generated token
          after the `decoder_start_token_id`. Useful for multilingual models like [mBART](../model_doc/mbart) where
          the first generated token needs to be the target language token.
        - **forced_eos_token_id** (`int`, *optional*) -- The id of the token to force as the last generated token when
          `max_length` is reached.
        - **remove_invalid_values** (`bool`, *optional*) -- Whether to remove possible _nan_ and _inf_ outputs of the
          model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down
          generation.

    Parameters for fine-tuning tasks:
        - **architectures** (`List[str]`, *optional*) -- Model architectures that can be used with the model
          pretrained weights.
        - **finetuning_task** (`str`, *optional*) -- Name of the task used to fine-tune the model. This can be used
          when converting from an original (TensorFlow or PyTorch) checkpoint.
        - **id2label** (`Dict[int, str]`, *optional*) -- A map from index (for instance prediction index, or target
          index) to label.
        - **label2id** (`Dict[str, int]`, *optional*) -- A map from label to index for the model.
        - **num_labels** (`int`, *optional*) -- Number of labels to use in the last layer added to the model,
          typically for a classification task.
        - **task_specific_params** (`Dict[str, Any]`, *optional*) -- Additional keyword arguments to store for the
          current task.
        - **problem_type** (`str`, *optional*) -- Problem type for `XxxForSequenceClassification` models. Can be one
          of (`"regression"`, `"single_label_classification"`, `"multi_label_classification"`).

    Parameters linked to the tokenizer:
        - **tokenizer_class** (`str`, *optional*) -- The name of the associated tokenizer class to use (if none is set,
          will use the tokenizer associated to the model by default).
        - **prefix** (`str`, *optional*) -- A specific prompt that should be added at the beginning of each text before
          tokenization.
        - **bos_token_id** (`int`, *optional*) -- The id of the _beginning-of-stream_ token.
        - **pad_token_id** (`int`, *optional*) -- The id of the _padding_ token.
        - **eos_token_id** (`int`, *optional*) -- The id of the _end-of-stream_ token.
        - **decoder_start_token_id** (`int`, *optional*) -- If an encoder-decoder model starts decoding with a
          different token than _bos_, the id of that token.
        - **sep_token_id** (`int`, *optional*) -- The id of the _separation_ token.

    PyTorch specific parameters:
        - **torchscript** (`bool`, *optional*, defaults to `False`) -- Whether or not the model should be used with
          Torchscript.
        - **tie_word_embeddings** (`bool`, *optional*, defaults to `True`) -- Whether the model's input and output word
          embeddings should be tied. Note that this is only relevant if the model has a output word embedding layer.
        - **torch_dtype** (`str`, *optional*) -- The `dtype` of the weights. This attribute can be used to initialize
          the model to a non-default `dtype` (which is normally `float32`) and thus allow for optimal storage and
          speed for inference. Please note that this only specifies the dtype of the computation and does not influence
          the dtype of model parameters.

        The list is non-exhaustive and specific parameters may apply to individual models. See the model's
        documentation for details on model-specific parameters.
    """

    model_type: str = ""
    is_composition: bool = False
    attribute_map: Dict[str, str] = {}
    _auto_class: Optional[str] = None

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)
        self.torch_dtype = kwargs.pop("torch_dtype", None)
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.tf_legacy_loss = kwargs.pop("tf_legacy_loss", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", True)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.cross_attention_hidden_size = kwargs.pop("cross_attention_hidden_size", None)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_scores = kwargs.pop("output_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)

        # Fine-tuning task specific parameters
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.id2label is not None:
            kwargs.pop("num_labels", None)
            self.id2label = dict((int(key), value) for key, value in self.id2label.items())
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)

        if self.label2id is not None and self.id2label is not None:
            assert len(self.label2id) == len(self.id2label)

        self.task_specific_params = kwargs.pop("task_specific_params", None)
        self.problem_type = kwargs.pop("problem_type", None)

        # Tokenizer kwargs
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific kwargs
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        hidden_size = kwargs.pop("hidden_size", None)
        if hidden_size is not None:
            self.hidden_size = hidden_size

        # Store model name/path if provided
        self.name_or_path = kwargs.pop("name_or_path", "")

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @property
    def name_or_path(self) -> str:
        return getattr(self, "_name_or_path", "")

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) -> int:
        """
        `int`: The number of labels for classification models.
        """
        return len(self.id2label) if self.id2label is not None else self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels: int):
        if not hasattr(self, "_num_labels"):
            self._num_labels = num_labels
        else:
            if self._num_labels != num_labels:
                logger.warning(
                    f"You are resizing the embedding layer from {self._num_labels} to {num_labels}, make sure this is"
                    " the intended behavior."
                )
            self._num_labels = num_labels

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, self.config_file_name)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "PretrainedConfig":
        r"""
        Instantiate a [`PretrainedConfig`] (or a derived class) from a pretrained model configuration.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~PretrainedConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
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

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        # We can't instantiate directly the base class *PretrainedConfig* so let's show the examples on a
        # derived class: BertConfig
        config = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased"
        )  # Download configuration from huggingface.co and cache.
        config = BertConfig.from_pretrained(
            "./test/saved_model/"
        )  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
        config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
        config = BertConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        assert config.output_attentions == True
        config, unused_kwargs = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        )
        assert config.output_attentions == True
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs.setdefault("cache_dir", cache_dir)
        kwargs.setdefault("force_download", force_download)
        kwargs.setdefault("local_files_only", local_files_only)
        kwargs.setdefault("revision", revision)

        cls._set_token_in_kwargs(kwargs, token)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PretrainedConfig`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_file_name)
        else:
            config_file = cls.config_file_name

        try:
            # Load from local folder or from cache or download from model Hub and cache
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )
            commit_hash = extract_commit_hash(resolved_config_file, commit_hash)

        except EnvironmentError:
            # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to the
            # original exception.
            raise
        except Exception:
            # For any other exception, we throw a generic error.
            raise EnvironmentError(
                f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
                " from 'https://huggingface.co/models', make sure you don't have a local directory with the same name."
                f" Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory"
                f" containing a {cls.config_file_name} file"
            )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {config_file} from cache at {resolved_config_file}")

        if "auto_map" in config_dict and not is_local:
            config_dict["auto_map"] = add_model_info_to_auto_map(
                config_dict["auto_map"], pretrained_model_name_or_path
            )
        if "custom_pipelines" in config_dict and not is_local:
            config_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                config_dict["custom_pipelines"], pretrained_model_name_or_path
            )

        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PretrainedConfig.get_config_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # We remove it from kwargs so that it does not appear in `return_unused_kwargs`.
        config_dict.update(kwargs)
        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

        # Update config with kwargs if needed
        if return_unused_kwargs:
            # Filter out None values in unused_kwargs by leveraging the fact that we can detect if each key is used
            # by seeing if it's in the config __dict__
            unused_kwargs = {key: value for key, value in kwargs.items() if key not in config.__dict__}
            logger.debug(f"The following `kwargs` are not used by the configuration and have been ignored: {unused_kwargs}")
            return config, unused_kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return isinstance(other, PretrainedConfig) and (self.__dict__ == other.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # For composition configs, we need to treat the nested configs as dicts
                diff = recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # Needs the model type to rebuild model
                    diff["model_type"] = value["model_type"]
                if len(diff) > 0:
                    serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = serializable_config_dict.pop("_pre_quantization_dtype", None)

        self.dict_torch_dtype_to_str(serializable_config_dict)

        if "_commit_hash" in serializable_config_dict:
            del serializable_config_dict["_commit_hash"]

        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

        for key, value in output.items():
            # Deal with nested configs like CLIP-Text config and CLIP-Vision config
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]

            output[key] = value

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = output.pop("_pre_quantization_dtype", None)

        self.dict_torch_dtype_to_str(output)

        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and hasattr(d["torch_dtype"], "split"):
            d["torch_dtype"] = d["torch_dtype"].split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    @classmethod
    def _set_token_in_kwargs(cls, kwargs, token=None):
        """Temporary method to deal with `token` and `use_auth_token`.

        This method is to avoid apply the same changes in all model config classes that overwrite `from_pretrained`.

        Need to clean up `use_auth_token` in a follow PR.
        """
        # Some model config classes like CLIP define their own `from_pretrained` without the new `token` argument.
        # Instead of defining `token` in each config class, we deal this in this method.
        if token is None:
            token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

    @property
    def config_file_name(self):
        return "config.json"


class VLAConfig(PretrainedConfig):
    """
    Configuration class for nanoVLA models.
    
    This is the configuration class to store the configuration of a [`nanoVLA`] model. It is used to instantiate a
    nanoVLA model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the nanoVLA-50M model.
    
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Args:
        action_dim (`int`, *optional*, defaults to 7):
            Dimension of the action space (e.g., 7 for 7-DOF robot arm).
        hidden_dim (`int`, *optional*, defaults to 768):
            Hidden dimension used throughout the model.
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the vision encoder.
        language_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the language model.
        fusion_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the fusion module.
        vlm_name (`str`, *optional*):
            Name of the underlying VLM being used.
        transfer_strategy (`str`, *optional*, defaults to `"fine_tune_all"`):
            Transfer learning strategy to use. Options: "freeze_backbone", "fine_tune_all", "progressive_unfreezing", 
            "adapter_layers", "distillation", "hybrid".
        normalize_actions (`bool`, *optional*, defaults to `True`):
            Whether to normalize actions to [-1, 1] range.
        action_bounds (`list`, *optional*):
            Min and max bounds for each action dimension. If provided, should be list of [min, max] pairs.
        
    Example:
    
    ```python
    >>> from nanovla import VLAConfig, nanoVLA
    
    >>> # Initializing a nanoVLA configuration
    >>> configuration = VLAConfig()
    
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = nanoVLA(configuration)
    
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    
    model_type = "nanovla"
    is_composition = True
    
    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 768,
        vision_config: Optional[Dict[str, Any]] = None,
        language_config: Optional[Dict[str, Any]] = None,
        fusion_config: Optional[Dict[str, Any]] = None,
        vlm_name: Optional[str] = None,
        transfer_strategy: str = "fine_tune_all",
        normalize_actions: bool = True,
        action_bounds: Optional[List[List[float]]] = None,
        **kwargs,
    ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.vlm_name = vlm_name
        self.transfer_strategy = transfer_strategy
        self.normalize_actions = normalize_actions
        self.action_bounds = action_bounds
        
        # Vision configuration
        if vision_config is None:
            vision_config = {}
        self.vision_config = vision_config
        
        # Language configuration  
        if language_config is None:
            language_config = {}
        self.language_config = language_config
        
        # Fusion configuration
        if fusion_config is None:
            fusion_config = {
                "type": "cross_attention",
                "num_heads": 8,
                "dropout": 0.1,
            }
        self.fusion_config = fusion_config
        
        super().__init__(**kwargs)
    
    @classmethod
    def from_vlm_config(
        cls,
        vlm_name: str,
        action_dim: int = 7,
        **kwargs
    ) -> "VLAConfig":
        """
        Create a VLAConfig from a VLM name with intelligent defaults.
        
        Args:
            vlm_name: Name of the VLM to base configuration on
            action_dim: Dimension of action space
            **kwargs: Additional configuration options
            
        Returns:
            VLAConfig instance configured for the specified VLM
        """
        # Import here to avoid circular imports
        from .registry import vlm_registry
        
        capabilities = vlm_registry.get_capabilities(vlm_name)
        if capabilities is None:
            logger.warning(f"No capabilities found for VLM {vlm_name}, using default config")
            return cls(action_dim=action_dim, vlm_name=vlm_name, **kwargs)
        
        # Set intelligent defaults based on VLM capabilities
        config_kwargs = {
            "action_dim": action_dim,
            "vlm_name": vlm_name,
            "transfer_strategy": capabilities.transfer_learning_strategy,
            **kwargs
        }
        
        # Set hidden dimension based on VLM feature dimensions
        if capabilities.feature_dimensions:
            vision_dim = capabilities.feature_dimensions.get("vision", 768)
            language_dim = capabilities.feature_dimensions.get("language", 768)
            config_kwargs["hidden_dim"] = max(vision_dim, language_dim)
        
        return cls(**config_kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        return output





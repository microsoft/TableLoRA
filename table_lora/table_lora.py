# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from imp import reload
import inspect
from typing import Any, List, Optional, Union
import warnings
import packaging
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.utils import PushToHubMixin
from peft import PeftModel, PeftModelForCausalLM
from peft.config import PeftConfig
from peft.utils.other import _get_batch_size
from peft.utils.peft_types import PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils import _prepare_prompt_learning_config, set_peft_model_state_dict, infer_device, load_peft_weights
from enum import Enum
import torch
import transformers
from peft.peft_model import PeftModel
from peft import PeftMixedModel, get_peft_model
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
import os
import torch.nn as nn
from peft.tuners.lora.layer import Linear, LoraLayer
from peft.tuners import XLoraModel, XLoraConfig
from accelerate.utils import get_balanced_memory, named_module_tensors

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
class TABLE_TOKEN(Enum):
    TABLE = "<TABLE>"
    ROW = "<ROW>"
    COL = "<COL>"

class PeftModel_new(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    **Attributes**:
        - **base_model** ([`torch.nn.Module`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
            saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
            using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
            using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
            backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
            in the base model if using [`PromptLearningConfig`].
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_configs: Union[List[PeftConfig],PeftConfig],
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
    ) -> None:
        # super().__init__()
        super(PeftModel, self).__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        # self.peft_type = peft_config.peft_type
        if isinstance(peft_configs, PeftConfig):
            peft_configs = [peft_configs]
        self.peft_type = [peft_config.peft_type for peft_config in peft_configs]
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        # self._is_prompt_learning = peft_config.is_prompt_learning
        # if self._is_prompt_learning:
        #     self._peft_config = {adapter_name: peft_config}
        #     self.base_model = model
        #     self.add_adapter(adapter_name, peft_config)
        # else:
        #     self._peft_config = None
        #     cls = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type]
        #     self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
        #     self.set_additional_trainable_modules(peft_config, adapter_name)

        self._is_prompt_learning = False
        self._is_adapter = False
        self.base_model = None
        self._peft_config = None
        for peft_config in peft_configs:
            if peft_config.is_prompt_learning:
                if self._is_prompt_learning:
                    raise ValueError("Only one prompt learning method can be used at a time.")
                self._is_prompt_learning = True
                self._peft_config = {adapter_name: peft_config}
                self.base_model = model if self.base_model is None else self.base_model
                self.add_adapter(adapter_name, peft_config)
            else:
                self._is_adapter = True
                cls = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type]
                if self.base_model is not None and self.base_model != model:
                    raise ValueError("Only one Peft adapter method can be used at a time.")
                adapter_name_ = adapter_name if len(peft_configs)==1 else f"{adapter_name}_1"
                self.base_model = cls(model, {adapter_name_: peft_config}, adapter_name_)
                self.set_additional_trainable_modules(peft_config, adapter_name_)

            if hasattr(self.base_model, "_cast_adapter_dtype"):
                self.base_model._cast_adapter_dtype(
                    adapter_name=adapter_name_, autocast_adapter_dtype=autocast_adapter_dtype
                )

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        This adapter is not trained. To load a trained adapter, check out [`PeftModel.load_adapter`].

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
        """
        # if peft_config.peft_type != self.peft_type:
        if peft_config.peft_type not in self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )

        try:
            if peft_config.is_prompt_learning:
                self.peft_config[adapter_name] = peft_config
                if hasattr(self.config, "to_dict"):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config

                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
            elif peft_config.is_adaption_prompt:
                self.base_model.add_adapter(adapter_name, peft_config)
            else:
                self.peft_config[adapter_name] = peft_config
                self.base_model.inject_adapter(self.base_model.model, adapter_name)
        except Exception:  # something went wrong, roll back
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

        self.set_additional_trainable_modules(peft_config, adapter_name)
    # @property
    # def peft_config(self) -> dict[str, PeftConfig]:
    #     if self._is_prompt_learning:
    #         return self._peft_config
    #     return self.base_model.peft_config
    def peft_config_getter(self) -> dict[str, PeftConfig]:
        peft_config = {}
        if self._is_prompt_learning:
            peft_config.update(self._peft_config)
        if self._is_adapter:
            peft_config.update(self.base_model.peft_config)
        return peft_config

    # @peft_config.setter
    # def peft_config(self, value: dict[str, PeftConfig]):
    #     if self._is_prompt_learning:
    #         self._peft_config = value
    #     else:
    #         self.base_model.peft_config = value
    def peft_config_setter(self, value: dict[str, PeftConfig]):
        for adapter_name, peft_config in value.items():
            if peft_config._is_prompt_learning:
                self._peft_config = {adapter_name: peft_config}
            else:
                self.base_model = {adapter_name: peft_config}

    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        # return (
        #     self.base_model
        #     if (self.active_peft_config.is_prompt_learning or self.peft_type == PeftType.POLY)
        #     else self.base_model.model
        # )
        return (
            self.base_model.model
            if (self._is_adapter and PeftType.POLY not in self.peft_type)
            else self.base_model
        )
    
    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        **kwargs: Any,
    ) -> PeftModel:
        r"""
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

        Note that the passed `model` may be modified inplace.

        Args:
            model ([`torch.nn.Module`]):
                The model to be adapted. For 🤗 Transformers models, the model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`].
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuration. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`. Only relevant for specific adapter types.
            ephemeral_gpu_offload (`bool`, *optional*):
                Whether to use ephemeral GPU offloading for partially loaded modules. Defaults to `False`. This is
                useful when parts of the model and/or components (such as adapters) are kept in CPU memory until they
                are needed. Rather than perform expensive operations on small data, the data is transferred to the GPU
                on-demand, the operation(s) performed, and the results moved back to CPU memory. This brings a slight
                momentary VRAM overhead but gives orders of magnitude speedup in certain cases.
            torch_device (`str`, *optional*, defaults to None):
                The device to load the adapter on. If `None`, the device will be inferred.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific PEFT configuration class.
        """
        from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        entries = os.listdir(model_id) 
        folders = ["."]
        folders.extend([entry for entry in entries if os.path.isdir(os.path.join(model_id, entry))])
        configs = []
        adapter_names = []
        for folder in folders:
            kwargs["subfolder"] = folder
            # load the config
            if config is None:
                try:
                    config = PEFT_TYPE_TO_CONFIG_MAPPING[
                        PeftConfig._get_peft_type(
                            model_id,
                            subfolder=kwargs.get("subfolder", None),
                            revision=kwargs.get("revision", None),
                            cache_dir=kwargs.get("cache_dir", None),
                            use_auth_token=kwargs.get("use_auth_token", None),
                            token=kwargs.get("token", None),
                        )
                    ].from_pretrained(model_id, **kwargs)
                except:
                    continue
            elif isinstance(config, PeftConfig):
                config.inference_mode = not is_trainable
            else:
                raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

            # Runtime configuration, if supported
            if hasattr(config, "runtime_config"):
                config.runtime_config.ephemeral_gpu_offload = ephemeral_gpu_offload
            else:
                if ephemeral_gpu_offload:
                    warnings.warn("Ephemeral GPU offloading is not supported for this model. Ignoring.")

            if config.is_prompt_learning and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                config.inference_mode = not is_trainable
            if isinstance(getattr(model, "base_model", None), XLoraModel):
                if not isinstance(config, XLoraConfig):
                    raise TypeError(f"Expected 'XLoraConfig', got '{type(config)}' instead.")
                if "adapters" in kwargs:
                    config.adapters = kwargs["adapters"]
                else:
                    # If the path is on HF hub, then we get the adapter names to create a subfolders list which tells
                    # `load_adapter` where the adapters are.
                    if not os.path.exists(model_id):
                        s = HfFileSystem()

                        # The names of the adapters which must be in folders
                        adapter_names = [
                            file["name"][len(model_id) + 1 :] for file in s.ls(model_id) if file["type"] == "directory"
                        ]
                        # Prepare a dict of adapter paths, which really just point to the hf id; we will use the subfolders
                        adapter_paths = {}
                        for adapter_name in adapter_names:
                            adapter_paths[adapter_name] = os.path.join(model_id, model_id)
                        config.adapters = adapter_paths
                        config._subfolders = adapter_names
                    else:
                        if "adapters" not in kwargs:
                            raise ValueError("If model_id is a local path, then `adapters` must be passed in kwargs.")
            configs.append(config)
            config = None
            adapter_names.append(folder if folder != "." else "default")
        

        if hasattr(model, "hf_device_map"):
            weight_map = dict(named_module_tensors(model, recurse=True))

            # recreate the offload_index for disk-offloaded modules: we need to know the location in storage of each weight
            # before the offload hook is removed from the model
            disk_modules = set()
            index = None
            for name, module in model.named_modules():
                if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "original_devices"):
                    if hasattr(module._hf_hook.weights_map, "dataset"):
                        index = module._hf_hook.weights_map.dataset.index
                    for key in module._hf_hook.original_devices.keys():
                        if module._hf_hook.original_devices[key] == torch.device("meta"):
                            disk_modules.add(str(name) + "." + str(key))

            if disk_modules and not kwargs.get("use_safetensors", True):
                raise ValueError("Disk offloading currently only supported for safetensors")

            if index:
                offload_index = {
                    p: {
                        "safetensors_file": index[p]["safetensors_file"],
                        "weight_name": p,
                        "dtype": str(weight_map[p].dtype).replace("torch.", ""),
                    }
                    for p in weight_map.keys()
                    if p in disk_modules
                }
                kwargs["offload_index"] = offload_index


        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if configs[0].task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, configs, adapter_name, autocast_adapter_dtype=autocast_adapter_dtype)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[configs[0].task_type](
                model, configs, adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        model.load_adapter(
            model_id, adapter_names, is_trainable=is_trainable, autocast_adapter_dtype=autocast_adapter_dtype, **kwargs
        )

        return model
    
    def load_adapter(
        self,
        model_id: str,
        adapter_name: Union[str, List[str]],
        is_trainable: bool = False,
        torch_device: Optional[str] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        **kwargs: Any,
    ):
        """
        Load a trained adapter into the model.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            torch_device (`str`, *optional*, defaults to None):
                The device to load the adapter on. If `None`, the device will be inferred.
            autocast_adapter_dtype (`bool`, *optional*, defaults to `True`):
                Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter
                weights using float16 and bfloat16 to float32, as this is typically required for stable training, and
                only affect select PEFT tuners.
            ephemeral_gpu_offload (`bool`, *optional*, defaults to `False`):
                Whether to use ephemeral GPU offloading for partially loaded modules. Defaults to `False`.
            kwargs: (`optional`):
                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
        """
        from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        if torch_device is None:
            torch_device = infer_device()

        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]
        
        for name in adapter_name:
            if name not in self.peft_config:
                hf_hub_download_kwargs["subfolder"] = name if name != "default" else None
                # load the config
                peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                    PeftConfig._get_peft_type(
                        model_id,
                        **hf_hub_download_kwargs,
                    )
                ].from_pretrained(
                    model_id,
                    ephemeral_gpu_offload=ephemeral_gpu_offload,
                    **hf_hub_download_kwargs,
                )
                if peft_config.is_prompt_learning and is_trainable:
                    raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
                else:
                    peft_config.inference_mode = not is_trainable
                self.add_adapter(name, peft_config)

        for name in adapter_name:
            hf_hub_download_kwargs["subfolder"] = name if name != "default" else None
            adapters_weights = load_peft_weights(model_id, device=torch_device, **hf_hub_download_kwargs)

            # load the weights into the model
            ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
            load_result = set_peft_model_state_dict(
                self, adapters_weights, adapter_name=name, ignore_mismatched_sizes=ignore_mismatched_sizes
            )

        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )

            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )

            self._update_offload(offload_index, adapters_weights)
            dispatch_model_kwargs["offload_index"] = offload_index

            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )

            hook = AlignDevicesHook(io_same_device=True)
            if self.is_prompt_learning:
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)


        for name in adapter_name:
            if hasattr(self.base_model, "_cast_adapter_dtype"):
                self.base_model._cast_adapter_dtype(
                    adapter_name=name, autocast_adapter_dtype=autocast_adapter_dtype
                )

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result


class PeftModelForCausalLM_new(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(
        self, model: torch.nn.Module, peft_config: Union[List[PeftConfig],PeftConfig], adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        row_ids=None,
        col_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        # if attention_mask is not None:
        #     # concat prompt attention mask
        #     prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
        #     attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
            )
        else:
            # if inputs_embeds is None:
            #     inputs_embeds = self.word_embeddings(input_ids)
            # # concat prompt labels
            # if labels is not None:
            #     prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
            #     kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            # prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            # prompts = prompts.to(inputs_embeds.dtype)
            # inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

            # For prompt tuning
            inputs_embeds = torch.zeros((input_ids.shape[0], input_ids.shape[1], self.word_embeddings.embedding_dim), device=input_ids.device)
            token_num = self.word_embeddings.num_embeddings
            mask = input_ids < token_num

            word_embeds = self.word_embeddings(input_ids[mask])

            prompt_encoder = self.prompt_encoder[self.active_adapter]
            if peft_config.inference_mode:
                prompt_tuning_emb = prompt_encoder.embedding((input_ids[~mask] - token_num))
            else:
                prompt_tuning_emb = prompt_encoder((input_ids[~mask] - token_num))

            prompt_tuning_emb = prompt_tuning_emb.to(word_embeds.dtype)
            inputs_embeds = inputs_embeds.to(word_embeds.dtype)
            inputs_embeds[mask] = word_embeds  
            inputs_embeds[~mask] = prompt_tuning_emb   

            # For embed lora
            self.row_ids = row_ids
            self.col_ids = col_ids
            for name, module in self.base_model.named_modules():  
                if isinstance(module, Linear):  
                    module.row_ids = row_ids
                    module.col_ids = col_ids

            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
    
    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.38, all architectures should support caching.
        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        uses_cache = uses_transformers_4_38 or (
            uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs["past_key_values"] is not None):
                # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
                # In prompt learning methods, past key values are longer when compared to the `input_ids`.
                # As such only consider the last input ids in the autogressive generation phase.
                past_key_values = model_kwargs["past_key_values"]
                if isinstance(past_key_values, (tuple, list)):
                    seq_len = past_key_values[0][0].shape[-2]
                else:  # using transformers kv cache
                    seq_len = past_key_values.get_seq_length()
                if seq_len >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

            # if model_kwargs.get("attention_mask", None) is not None:
            #     size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
            #     prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
            #     model_kwargs["attention_mask"] = torch.cat(
            #         (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
            #     )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            # else:
            #     if model_kwargs["past_key_values"] is None:
            #         inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
            #         prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
            #         prompts = prompts.to(inputs_embeds.dtype)
            #         model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
            #         model_kwargs["input_ids"] = None
            elif peft_config.peft_type != PeftType.PREFIX_TUNING:
                # For embed lora
                if self.row_ids is None:
                    self.row_ids = torch.zeros(model_kwargs["input_ids"].shape[:2], dtype=torch.long).to(model_kwargs["input_ids"].device)
                    self.col_ids = torch.zeros(model_kwargs["input_ids"].shape[:2], dtype=torch.long).to(model_kwargs["input_ids"].device)
                for name, module in self.get_base_model().named_modules():  
                    if isinstance(module, Linear):  
                        module.row_ids = self.row_ids
                        module.col_ids = self.col_ids
                self.row_ids = None
                self.col_ids = None

                input_ids = model_kwargs["input_ids"]
                inputs_embeds = torch.zeros((input_ids.shape[0], input_ids.shape[1], self.word_embeddings.embedding_dim), device=input_ids.device)
                token_num = self.word_embeddings.num_embeddings
                mask = input_ids < token_num

                word_embeds = self.word_embeddings(input_ids[mask])

                prompt_encoder = self.prompt_encoder[self.active_adapter]
                prompt_tuning_emb = prompt_encoder.embedding((input_ids[~mask] - token_num))

                prompt_tuning_emb = prompt_tuning_emb.to(word_embeds.dtype)
                inputs_embeds = inputs_embeds.to(word_embeds.dtype)
                inputs_embeds[mask] = word_embeds  
                inputs_embeds[~mask] = prompt_tuning_emb

                model_kwargs["inputs_embeds"] = inputs_embeds
                model_kwargs["input_ids"] = None

        # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
        # passed in the forward pass to keep track of the position ids of the cache. We have to
        # pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
        # `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
        _ = model_kwargs.pop("cache_position", None)

        return model_kwargs
    
    def generate(self, *args, **kwargs):
        peft_config = self.active_peft_config
        # self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.get_base_model().prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(*args, **kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(*args, **kwargs)
            else:
                self.row_ids = kwargs.get("row_ids", None)
                self.col_ids = kwargs.get("col_ids", None)
                del kwargs["col_ids"]
                del kwargs["row_ids"]
                outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

from peft.tuners.tuners_utils import BaseTunerLayer
import math
from transformers.pytorch_utils import Conv1D
class LoraLayer_new(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", 
                           "lora_A_tab_col", "lora_A_tab_row", "lora_B_tab")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        # tab_lora_laryer = [".0.",".1.",".2.",".3.",".4.",".5.",".6.",".7.",]
        # is_tab_lora = False
        # for i in tab_lora_laryer:
        #     if i in kwargs["current_key"]:
        #         is_tab_lora = True
        #         break
        is_tab_lora = True

        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_dropout_tab = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        if is_tab_lora:
            self.scaling_tab = {}
            self.lora_A_tab_col = nn.ModuleDict({})
            self.lora_A_tab_row = nn.ModuleDict({})
            self.lora_B_tab = nn.ModuleDict({})
        self.is_tab_lora = is_tab_lora
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        if self.is_tab_lora:
            # self.lora_A_tab_col[adapter_name] = nn.Embedding(30, r)
            # self.lora_A_tab_row[adapter_name] = nn.Embedding(600, r)
            # For hitab: 
            # Max_col:  29, Max_row:  67, Train dataset (7417)
            # Max_col:  21, Max_row:  65, Test dataset (1584)
            # self.lora_A_tab_col[adapter_name] = nn.Embedding(40, r) 
            # self.lora_A_tab_row[adapter_name] = nn.Embedding(100, r)
            # For multihiertt:
            # Max_col:  17, Max_row:  66, Train dataset (7830)
            # Max_col:  14, Max_row:  62, Test dataset (1044)
            # self.lora_A_tab_col[adapter_name] = nn.Embedding(30, r) 
            # self.lora_A_tab_row[adapter_name] = nn.Embedding(100, r)
            # For turl:
            # Max_col:  38, Max_row:  314, Train dataset (62882)
            # Max_col:  15, Max_row:  176, Test dataset (2069)
            # self.lora_A_tab_col[adapter_name] = nn.Embedding(40, r)
            # self.lora_A_tab_row[adapter_name] = nn.Embedding(400, r)
            ## For all
            # self.lora_A_tab_col[adapter_name] = nn.Embedding(40, r)
            # self.lora_A_tab_row[adapter_name] = nn.Embedding(600, r)
            ## For tabfact
            self.lora_A_tab_col[adapter_name] = nn.Embedding(50, r)
            self.lora_A_tab_row[adapter_name] = nn.Embedding(600, r)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if self.is_tab_lora:
            self.lora_B_tab[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
            if self.is_tab_lora:
                self.scaling_tab[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
            if self.is_tab_lora:
                self.scaling_tab[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)
    
    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
                if self.is_tab_lora:
                    nn.init.kaiming_uniform_(self.lora_A_tab_col[adapter_name].weight, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.lora_A_tab_row[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
                if self.is_tab_lora:
                    nn.init.normal_(self.lora_A_tab_col[adapter_name].weight, std=1 / self.r[adapter_name])
                    nn.init.normal_(self.lora_A_tab_row[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            if self.is_tab_lora:
                nn.init.zeros_(self.lora_B_tab[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

class Linear_new(nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super(Linear, self).__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.row_embeds = None
        self.col_embeds = None


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                if self.is_tab_lora:
                    lora_A_tab_col = self.lora_A_tab_col[active_adapter]
                    lora_A_tab_row = self.lora_A_tab_row[active_adapter]
                lora_B = self.lora_B[active_adapter]
                if self.is_tab_lora:
                    lora_B_tab = self.lora_B_tab[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                if self.is_tab_lora:
                    scaling_tab = self.scaling_tab[active_adapter]
                x = x.to(lora_A.weight.dtype)
                row_ids = self.row_ids
                col_ids = self.col_ids

                if not self.use_dora[active_adapter]:
                    if self.is_tab_lora:
                        col_embeds = lora_A_tab_col(col_ids)
                        row_embeds = lora_A_tab_row(row_ids)
                        result = result + lora_B(lora_A(dropout(x))) * scaling + lora_B_tab(col_embeds + row_embeds) * scaling_tab
                    else:
                        result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result

from peft.tuners.lora.model import LoraModel
from peft.tuners.tuners_utils import BaseTuner
from itertools import chain
import re
from peft.utils import get_quantization_config
class LoraModel_new(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import torch
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        >>> rank = ...
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )
        >>> quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

        >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     bos_token="[BOS]",
        ...     eos_token="[EOS]",
        ...     unk_token="[UNK]",
        ...     pad_token="[PAD]",
        ...     mask_token="[MASK]",
        ... )
        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     quantization_config=quantization_config,
        ... )
        >>> model = prepare_model_for_kbit_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name) -> None:
        super(LoraModel, self).__init__(model, config, adapter_name)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "current_key": current_key,
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)


def get_peft_model(
    model: PreTrainedModel,
    peft_config: Union[List[PeftConfig], PeftConfig],
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 or bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        revision (`str`, `optional`, defaults to `main`):
            The revision of the base model. If this isn't set, the saved peft model will load the `main` revision for
            the base model
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    # peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    # if mixed:
    #     # note: PeftMixedModel does not support autocast_adapter_dtype, so don't pass it
    #     return PeftMixedModel(model, peft_config, adapter_name=adapter_name)

    # if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
    #     return PeftModel(model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype)

    # if peft_config.is_prompt_learning:
    #     peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    # return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
    #     model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
    # )
    if isinstance(peft_config, PeftConfig): # TODO: What about mixed?
        peft_config = [peft_config]
    for i in range(len(peft_config)):
        peft_config[i].base_model_name_or_path = model.__dict__.get("name_or_path", None)
        if peft_config[i].is_prompt_learning:
            peft_config[i] = _prepare_prompt_learning_config(peft_config[i], model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config[0].task_type](model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype)

from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import numpy as np
from transformers import DataCollatorForSeq2Seq
class DataCollatorForSeq2Seq_new:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        # if labels is not None:
        #     max_label_length = max(len(l) for l in labels)
        #     if self.pad_to_multiple_of is not None:
        #         max_label_length = (
        #             (max_label_length + self.pad_to_multiple_of - 1)
        #             // self.pad_to_multiple_of
        #             * self.pad_to_multiple_of
        #         )

        #     padding_side = self.tokenizer.padding_side
        #     for feature in features:
        #         remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
        #         if isinstance(feature["labels"], list):
        #             feature["labels"] = (
        #                 feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
        #             )
        #         elif padding_side == "right":
        #             feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
        #         else:
        #             feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        for key in features[0].keys():
            if key in ["input_ids", "attention_mask"]:
                continue
            pad_value = self.label_pad_token_id if key == "labels" else 0
            labels = [feature[key] for feature in features] if key in features[0].keys() else None
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
            # same length to return tensors.
            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [pad_value] * (max_label_length - len(feature[key]))
                    if isinstance(feature[key], list):
                        feature[key] = (
                            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
                        )
                    elif padding_side == "right":
                        feature[key] = np.concatenate([feature[key], remainder]).astype(np.int64)
                    else:
                        feature[key] = np.concatenate([remainder, feature[key]]).astype(np.int64)

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

def load_table_lora():
    # get_peft_model = get_peft_model_new
    LoraModel.__init__ = LoraModel_new.__init__
    LoraModel._create_and_replace = LoraModel_new._create_and_replace
    PeftModel.__init__ = PeftModel_new.__init__
    PeftModel.peft_config = property(PeftModel_new.peft_config_getter, PeftModel_new.peft_config_setter)
    PeftModel.add_adapter = PeftModel_new.add_adapter
    PeftModel.get_base_model = PeftModel_new.get_base_model
    PeftModel.from_pretrained = PeftModel_new.from_pretrained
    PeftModel.load_adapter = PeftModel_new.load_adapter
    PeftModelForCausalLM.forward = PeftModelForCausalLM_new.forward
    PeftModelForCausalLM.prepare_inputs_for_generation = PeftModelForCausalLM_new.prepare_inputs_for_generation
    PeftModelForCausalLM.generate = PeftModelForCausalLM_new.generate
    DataCollatorForSeq2Seq.__call__ = DataCollatorForSeq2Seq_new.__call__
    Linear.forward = Linear_new.forward
    Linear.__init__ = Linear_new.__init__ 
    LoraLayer.__init__ = LoraLayer_new.__init__
    LoraLayer.update_layer = LoraLayer_new.update_layer
    LoraLayer.reset_lora_parameters = LoraLayer_new.reset_lora_parameters
    # import inspect
    # source_code = inspect.getsource(PeftModel.__init__)
    # print(source_code)
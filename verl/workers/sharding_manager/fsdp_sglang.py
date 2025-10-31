# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import logging
import os

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.debug.performance import _timer
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.model import convert_weight_keys
from verl.utils.torch_functional import check_device_is_available

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with _timer("reshard", self.timing):
            torch.cuda.empty_cache()
            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)
            params = self.module.state_dict()
            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
            device = torch.cuda.current_device()  # used when fsdp2 set cpu_offload_policy
            params = {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
            params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
            
            # Handle LoRA parameter name conversion for SGLang
            unwrapped_module = getattr(self.module, "_fsdp_wrapped_module", self.module)
            if hasattr(unwrapped_module, "peft_config"):
                # Model has LoRA enabled, need to merge LoRA weights into base model
                logger.info("LoRA detected, will merge LoRA weights into base model for SGLang")
                
                # Get LoRA config for proper scaling
                peft_config = unwrapped_module.peft_config
                if peft_config:
                    # Get the first adapter config (usually "default")
                    adapter_name = list(peft_config.keys())[0]
                    config = peft_config[adapter_name]
                    self._lora_scaling = config.lora_alpha / config.r
                    logger.info(f"LoRA scaling factor: {self._lora_scaling:.2f} (alpha={config.lora_alpha}, rank={config.r})")
                else:
                    self._lora_scaling = 2.0  # Fallback
                
                params = self._convert_lora_params_to_base_names(params)
            # loop = asyncio.get_event_loop()
            # NOTE: try catch to avoid the event loop is not initialized error raised in calling _async_rollout_a_request() inside an _async_rollout_a_batch() in the multi-agent setting
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(self.update_weights(params))
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            torch.cuda.empty_cache()
            log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        # loop = asyncio.get_event_loop()
        # NOTE: try catch to avoid the event loop is not initialized error raised in calling _async_rollout_a_request() inside an _async_rollout_a_batch() in the multi-agent setting
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self.release_memory())
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
            
    def _convert_lora_params_to_base_names(self, params):
        """
        Convert LoRA parameter names to base model names for SGLang.
        
        Has two modes:
        1. If LoRA adapters present: Manually merge LoRA (A @ B) into base weights
        2. If only base weights: Just convert names
        
        This ensures SGLang uses the complete trained model.
        """
        converted_params = {}
        lora_params = {}
        
        # First pass: separate base params and LoRA params
        for name, param in params.items():
            if "lora_A" in name or "lora_B" in name:
                lora_params[name] = param
            elif "lora_embedding_" in name or "modules_to_save" in name:
                continue  # Skip these
            else:
                # This is a base layer parameter
                clean_name = name
                if clean_name.startswith("base_model.model."):
                    clean_name = clean_name[len("base_model.model."):]
                clean_name = clean_name.replace(".base_layer.", ".")
                converted_params[clean_name] = param
        
        # Second pass: merge LoRA weights into base weights
        if lora_params:
            logger.info(f"Found {len(lora_params)} LoRA parameters, performing manual merge")
            merged_count = 0
            
            # Group LoRA params by layer
            lora_groups = {}
            for name, param in lora_params.items():
                # Extract base name (remove lora_A/lora_B.default.weight suffix)
                if "lora_A" in name:
                    base_name = name.replace(".lora_A.default.weight", "")
                elif "lora_B" in name:
                    base_name = name.replace(".lora_B.default.weight", "")
                else:
                    continue
                
                if base_name not in lora_groups:
                    lora_groups[base_name] = {}
                
                if "lora_A" in name:
                    lora_groups[base_name]['A'] = param
                else:
                    lora_groups[base_name]['B'] = param
            
            # Merge each LoRA adapter into corresponding base weight
            for base_name, lora_pair in lora_groups.items():
                if 'A' not in lora_pair or 'B' not in lora_pair:
                    continue
                
                # Convert PEFT name to base model name
                clean_name = base_name
                if clean_name.startswith("base_model.model."):
                    clean_name = clean_name[len("base_model.model."):]
                clean_name = clean_name.replace(".base_layer", "")
                clean_name = clean_name + ".weight"  # LoRA only applies to weights
                
                if clean_name in converted_params:
                    # Merge: W = W_base + (B @ A) * scaling
                    # scaling = lora_alpha / lora_rank
                    lora_A = lora_pair['A']
                    lora_B = lora_pair['B']
                    
                    # Use configured scaling factor
                    scaling = getattr(self, '_lora_scaling', 2.0)
                    
                    # Compute delta: B @ A  
                    # LoRA formula: W = W_base + scaling * (B @ A)
                    delta = torch.matmul(lora_B, lora_A) * scaling
                    
                    # Add delta to base weight (in-place to save memory)
                    converted_params[clean_name] = converted_params[clean_name] + delta
                    merged_count += 1
            
            logger.info(f"Merged {merged_count} LoRA adapters into base model weights for SGLang")
        
        logger.info(f"Prepared {len(converted_params)} parameters for SGLang (LoRA merged: {len(lora_params) > 0})")
        return converted_params
    
    async def update_weights(self, params):
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self.inference_engine.resume_memory_occupation()

        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        named_tensors = [(k, v) for k, v in params.items()]
        load_format = None
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                gathered_serialized_tensors = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                await self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=tensor_index == len(named_tensors) - 1,
                )

    async def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        torch.cuda.empty_cache()
        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        device = torch.cuda.current_device()  # used when fsdp2 set cpu_offload_policy
        params = {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
        
        # Handle LoRA parameter name conversion for SGLang
        unwrapped_module = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(unwrapped_module, "peft_config"):
            logger.info("LoRA detected in wake_up, will merge LoRA weights for SGLang")
            
            # Get LoRA config for proper scaling
            peft_config = unwrapped_module.peft_config
            if peft_config:
                adapter_name = list(peft_config.keys())[0]
                config = peft_config[adapter_name]
                self._lora_scaling = config.lora_alpha / config.r
            else:
                self._lora_scaling = 2.0
            
            params = self._convert_lora_params_to_base_names(params)
        
        # Copy, not share memory
        await self.update_weights(params)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        del params
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        await self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

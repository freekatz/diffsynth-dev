import torch
from typing import List
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, PipelineUnit, WanVideoUnit_PromptEmbedder
from diffsynth.models.wan_video_4d_dit import Wan4DModel


class WanVideoUnit_4DConditionPreparation(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"
            ),
            output_params=("condition_latents", "condition_mask", "temporal_coords", "plucker_embedding", "num_views"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, num_frames, height, width, tiled, tile_size, tile_stride, **kwargs):
        return {
            "temporal_coords": pipe._temp_temporal_coords,
            "plucker_embedding": pipe._temp_plucker_embedding,
            "num_views": pipe._temp_num_views,
            "condition_latents": pipe._temp_condition_latents.to(dtype=pipe.torch_dtype, device=pipe.device),
            "condition_mask": pipe._temp_condition_mask.to(dtype=pipe.torch_dtype, device=pipe.device),
        }


class WanVideoUnit_4DPromptContextOverride(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        prompt_context = getattr(pipe, "_temp_prompt_context", None)
        if prompt_context is not None:
            inputs_posi["context"] = prompt_context.to(device=pipe.device, dtype=pipe.torch_dtype)
        return inputs_shared, inputs_posi, inputs_nega


def model_fn_wan4d_video(
    dit, latents, timestep, context, y=None, clip_feature=None,
    condition_latents=None, condition_mask=None, temporal_coords=None,
    plucker_embedding=None, num_views=2,
    use_unified_sequence_parallel=False, use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    return dit(
        x=latents,
        timestep=timestep,
        context=context,
        temporal_coords=temporal_coords,
        plucker_embedding=plucker_embedding,
        clip_feature=clip_feature,
        condition_latents=condition_latents,
        condition_mask=condition_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        num_views=num_views,
    )


class Wan4DPipeline(WanVideoPipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.units.insert(1, WanVideoUnit_4DConditionPreparation())
        prompt_idx = next(
            (i for i, unit in enumerate(self.units) if isinstance(unit, WanVideoUnit_PromptEmbedder)),
            None,
        )
        if prompt_idx is not None:
            self.units.insert(prompt_idx + 1, WanVideoUnit_4DPromptContextOverride())
        self.model_fn = model_fn_wan4d_video

    @staticmethod
    def from_pretrained(model_configs, tokenizer_config=None, device="cuda", torch_dtype=torch.bfloat16, **kwargs):
        base_pipe = WanVideoPipeline.from_pretrained(
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        pipe = Wan4DPipeline(device=device, torch_dtype=torch_dtype)
        pipe.text_encoder = base_pipe.text_encoder
        pipe.tokenizer = base_pipe.tokenizer
        pipe.image_encoder = base_pipe.image_encoder
        pipe.vae = base_pipe.vae
        pipe.scheduler = base_pipe.scheduler
        pipe.vram_management_enabled = base_pipe.vram_management_enabled
        pipe.dit = Wan4DModel.from_wan_model(base_pipe.dit)
        return pipe

    def __call__(
        self,
        *args,
        condition_latents: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        temporal_coords: torch.Tensor = None,
        plucker_embedding: torch.Tensor = None,
        prompt_context: torch.Tensor = None,
        num_views: int = 2,
        **kwargs
    ):
        self._temp_condition_latents = condition_latents
        self._temp_condition_mask = condition_mask
        self._temp_temporal_coords = temporal_coords
        self._temp_plucker_embedding = plucker_embedding
        self._temp_prompt_context = prompt_context
        self._temp_num_views = num_views
        return super().__call__(*args, **kwargs)

import torch
from typing import Optional, List, Union
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, PipelineUnit, WanVideoUnit_PromptEmbedder
from diffsynth.models.wan_video_4d_dit import Wan4DModel

class WanVideoUnit_4DConditionPreparation(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"
            ),
            output_params=("condition_latents", "condition_mask", "temporal_coords", "camera_embedding"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, num_frames, height, width, tiled, tile_size, tile_stride, **kwargs):
        temporal_coords = getattr(pipe, "_temp_temporal_coords", None)
        camera_embedding = getattr(pipe, "_temp_camera_embedding", None)
        output = {"temporal_coords": temporal_coords, "camera_embedding": camera_embedding}

        # Mode A: pre-computed condition tensors
        condition_latents = getattr(pipe, "_temp_condition_latents", None)
        condition_mask = getattr(pipe, "_temp_condition_mask", None)
        if condition_latents is not None and condition_mask is not None:
            output["condition_latents"] = condition_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            output["condition_mask"] = condition_mask.to(dtype=pipe.torch_dtype, device=pipe.device)
            return output

        # Mode B: sparse image references — encode each PIL Image via the VAE.
        reference_frames = getattr(pipe, "_temp_reference_frames", None)
        reference_indices = getattr(pipe, "_temp_reference_indices", None)
        reference_unit_ranges = getattr(pipe, "_temp_reference_unit_ranges", None)

        if reference_frames is not None and reference_indices is not None:
            pipe.load_models_to_device(self.onload_model_names)

            F_latent = (num_frames - 1) // 4 + 1
            H_l = height // 8
            W_l = width // 8

            # 构建条件视频 [1, C, T, H, W] 和像素空间 mask [1, T, H_l, W_l]
            cond_video = torch.zeros(
                (1, 3, num_frames, height, width),
                dtype=pipe.torch_dtype, device=pipe.device,
            )
            pixel_mask = torch.zeros(
                (1, num_frames, H_l, W_l),
                dtype=pipe.torch_dtype, device=pipe.device,
            )

            for ref_img, idx in zip(reference_frames, reference_indices):
                if idx >= num_frames:
                    continue
                ref_img = ref_img.resize((width, height))
                # preprocess_video 返回 [1, C, 1, H, W]，取第 0 帧得 [C, H, W]
                frame_t = pipe.preprocess_video([ref_img])[0, :, 0]
                cond_video[0, :, idx] = frame_t
                pixel_mask[0, idx] = 1.0

            condition_latents = pipe.vae.encode(
                cond_video, device=pipe.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

            # Time-fold: replicate first frame mask × 4 to align with VAE chunk-0 causal conv
            mask_folded = torch.cat([
                pixel_mask[:, 0:1].expand(-1, 4, -1, -1),
                pixel_mask[:, 1:],
            ], dim=1)
            condition_mask = mask_folded.view(1, F_latent, 4, H_l, W_l).permute(0, 2, 1, 3, 4).contiguous()

            output["condition_latents"] = condition_latents
            output["condition_mask"] = condition_mask

        return output

class WanVideoUnit_4DPromptContextOverride(PipelineUnit):
    """Replace positive text-encoder context with pre-computed prompt embeddings when available."""
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
    camera_embedding=None,
    use_unified_sequence_parallel=False, use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    if condition_latents is None and y is not None:
        if y.shape[1] == 20:
            condition_mask = y[:, 0:4]
            condition_latents = y[:, 4:20]
        else:
            condition_latents = y

    B = latents.shape[0]
    if condition_latents is not None and condition_latents.shape[0] != B:
        condition_latents = condition_latents.expand(B, -1, -1, -1, -1)
    if condition_mask is not None and condition_mask.shape[0] != B:
        condition_mask = condition_mask.expand(B, -1, -1, -1, -1)

    if temporal_coords is not None:
        if not isinstance(temporal_coords, torch.Tensor):
            temporal_coords = torch.tensor(temporal_coords, dtype=latents.dtype, device=latents.device)
        else:
            temporal_coords = temporal_coords.to(device=latents.device, dtype=latents.dtype)
        if temporal_coords.ndim == 1:
            temporal_coords = temporal_coords.unsqueeze(0).expand(B, -1)

    if camera_embedding is not None:
        if not isinstance(camera_embedding, torch.Tensor):
            camera_embedding = torch.tensor(camera_embedding, dtype=latents.dtype, device=latents.device)
        else:
            camera_embedding = camera_embedding.to(device=latents.device, dtype=latents.dtype)
        if camera_embedding.ndim == 2:
            camera_embedding = camera_embedding.unsqueeze(0).expand(B, -1, -1)

    x = dit(
        x=latents,
        timestep=timestep,
        context=context,
        temporal_coords=temporal_coords,
        camera_embedding=camera_embedding,
        clip_feature=clip_feature,
        condition_latents=condition_latents,
        condition_mask=condition_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    return x

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
    def from_pretrained(
        model_configs=None,
        tokenizer_config=None,
        device="cuda",
        torch_dtype=torch.bfloat16,
        **kwargs,
    ):
        """Load a pretrained Wan2.1 base model and wrap it as Wan4DPipeline with Wan4DModel dit."""
        if model_configs is None:
            model_configs = []
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
        if base_pipe.dit is not None:
            pipe.dit = Wan4DModel.from_wan_model(base_pipe.dit)
        return pipe

    def __call__(
        self,
        *args,
        reference_frames: Optional[List[Image.Image]] = None,
        reference_indices: Optional[List[int]] = None,
        reference_unit_ranges: Optional[List[tuple]] = None,
        condition_latents: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        temporal_coords: Optional[List[float]] = None,
        camera_embedding: Optional[torch.Tensor] = None,
        prompt_context: Optional[torch.Tensor] = None,
        **kwargs
    ):
        self._temp_reference_frames = reference_frames
        self._temp_reference_indices = reference_indices
        self._temp_reference_unit_ranges = reference_unit_ranges
        self._temp_condition_latents = condition_latents
        self._temp_condition_mask = condition_mask
        self._temp_temporal_coords = temporal_coords
        self._temp_camera_embedding = camera_embedding
        self._temp_prompt_context = prompt_context
        return super().__call__(*args, **kwargs)

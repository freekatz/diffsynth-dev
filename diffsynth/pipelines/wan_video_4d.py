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
            output_params=("condition_latents", "condition_mask", "temporal_coords"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, num_frames, height, width, tiled, tile_size, tile_stride, **kwargs):
        temporal_coords = getattr(pipe, "_temp_temporal_coords", None)
        output = {"temporal_coords": temporal_coords}

        # Mode A: pre-computed condition tensors (e.g. overfit-verification from target latent)
        condition_latents = getattr(pipe, "_temp_condition_latents", None)
        condition_mask = getattr(pipe, "_temp_condition_mask", None)
        if condition_latents is not None and condition_mask is not None:
            output["condition_latents"] = condition_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            output["condition_mask"] = condition_mask.to(dtype=pipe.torch_dtype, device=pipe.device)
            return output

        # Mode B: sparse image references — encode each PIL Image via the VAE using the
        # 4-frame trick, then fill the entire unit latent range with the encoded frame.
        reference_frames = getattr(pipe, "_temp_reference_frames", None)
        reference_indices = getattr(pipe, "_temp_reference_indices", None)
        # reference_unit_ranges: list of (latent_start, latent_end) per reference frame.
        # When provided, fills entire unit range instead of a single latent frame.
        reference_unit_ranges = getattr(pipe, "_temp_reference_unit_ranges", None)

        if reference_frames is not None and reference_indices is not None:
            pipe.load_models_to_device(self.onload_model_names)

            F_latent = (num_frames - 1) // 4 + 1
            H_l = height // 8
            W_l = width // 8

            # All-zero canvas for condition features and mask (batch_size=1; expanded in model_fn)
            condition_latents = torch.zeros((1, 16, F_latent, H_l, W_l), dtype=pipe.torch_dtype, device=pipe.device)
            condition_mask = torch.zeros((1, 1, F_latent, H_l, W_l), dtype=pipe.torch_dtype, device=pipe.device)

            for i, (ref_img, idx) in enumerate(zip(reference_frames, reference_indices)):
                ref_img = ref_img.resize((width, height))
                # Extract single pure Image latent using Video VAE
                # We replicate the image 4 times to trick the 3D VAE into producing exactly 1 pure stable latent slot
                img_tensor = pipe.preprocess_video([ref_img] * 4)
                z_i = pipe.vae.encode(img_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                z_i = z_i.to(dtype=pipe.torch_dtype, device=pipe.device) # shape: [1, 16, 1, H_l, W_l]

                target_latent_idx = idx // 4
                if target_latent_idx >= F_latent:
                    continue

                if reference_unit_ranges is not None and i < len(reference_unit_ranges):
                    # Fill entire unit latent range with this condition latent frame
                    lat_start, lat_end = reference_unit_ranges[i]
                    lat_start = max(0, min(lat_start, F_latent - 1))
                    lat_end = max(lat_start, min(lat_end, F_latent - 1))
                    condition_latents[:, :, lat_start:lat_end + 1, :, :] = z_i.expand(-1, -1, lat_end - lat_start + 1, -1, -1)
                    condition_mask[:, :, lat_start:lat_end + 1, :, :] = 1.0
                else:
                    # Fallback: fill only the single latent frame at the reference index
                    condition_latents[:, :, target_latent_idx:target_latent_idx+1, :, :] = z_i
                    condition_mask[:, :, target_latent_idx:target_latent_idx+1, :, :] = 1.0

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
    use_unified_sequence_parallel=False, use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    # If the standard 32-channel I2V is used via `y` but there's no explicitly passed Inpaint latents,
    # map `y` back for the `Wan4DModel` fallback to process.
    if condition_latents is None and y is not None:
        condition_latents = y

    # Expand condition tensors to match the actual batch size (e.g. CFG doubles the batch)
    B = latents.shape[0]
    if condition_latents is not None and condition_latents.shape[0] != B:
        condition_latents = condition_latents.expand(B, -1, -1, -1, -1)
    if condition_mask is not None and condition_mask.shape[0] != B:
        condition_mask = condition_mask.expand(B, -1, -1, -1, -1)

    # Normalize temporal_coords to (B, F_latent) tensor
    if temporal_coords is not None:
        if not isinstance(temporal_coords, torch.Tensor):
            temporal_coords = torch.tensor(temporal_coords, dtype=latents.dtype, device=latents.device)
        if temporal_coords.ndim == 1:
            temporal_coords = temporal_coords.unsqueeze(0).expand(B, -1)

    x = dit(
        x=latents,
        timestep=timestep,
        context=context,
        temporal_coords=temporal_coords,
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
        # Add the 4D Preprocessing Unit
        self.units.insert(1, WanVideoUnit_4DConditionPreparation())
        # Insert prompt-context override right after the PromptEmbedder so that
        # pre-computed text embeddings (when available) replace the encoded prompt.
        prompt_idx = next(
            (i for i, unit in enumerate(self.units) if isinstance(unit, WanVideoUnit_PromptEmbedder)),
            None,
        )
        if prompt_idx is not None:
            self.units.insert(prompt_idx + 1, WanVideoUnit_4DPromptContextOverride())
        # Hook the new model execution function
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
        # Load base WanVideoPipeline with pretrained weights
        base_pipe = WanVideoPipeline.from_pretrained(
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        # Create a properly configured Wan4DPipeline (units + model_fn)
        pipe = Wan4DPipeline(device=device, torch_dtype=torch_dtype)
        # Transfer model components from the base pipeline
        pipe.text_encoder = base_pipe.text_encoder
        pipe.tokenizer = base_pipe.tokenizer
        pipe.image_encoder = base_pipe.image_encoder
        pipe.vae = base_pipe.vae
        pipe.scheduler = base_pipe.scheduler
        pipe.vram_management_enabled = base_pipe.vram_management_enabled
        # Upgrade WanModel dit to Wan4DModel (adds mask_embedding, keeps all pretrained weights)
        if base_pipe.dit is not None:
            pipe.dit = Wan4DModel.from_wan_model(base_pipe.dit)
        return pipe

    def __call__(
        self,
        *args,
        reference_frames: Optional[List[Image.Image]] = None,
        reference_indices: Optional[List[int]] = None,      # Corresponding frame indices (pixel-space)
        reference_unit_ranges: Optional[List[tuple]] = None, # (latent_start, latent_end) per reference
        condition_latents: Optional[torch.Tensor] = None,   # Mode A: pre-computed condition latents (B,16,F,H,W)
        condition_mask: Optional[torch.Tensor] = None,      # Mode A: pre-computed condition mask (B,1,F,H,W)
        temporal_coords: Optional[List[float]] = None,
        prompt_context: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # We store arguments for our custom PipelineUnit to grab
        self._temp_reference_frames = reference_frames
        self._temp_reference_indices = reference_indices
        self._temp_reference_unit_ranges = reference_unit_ranges
        self._temp_condition_latents = condition_latents
        self._temp_condition_mask = condition_mask
        self._temp_temporal_coords = temporal_coords
        self._temp_prompt_context = prompt_context
        return super().__call__(*args, **kwargs)

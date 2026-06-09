# Image-To-Image

This example edits an input image with `Qwen/Qwen-Image-Edit` using the `image_edit.py` CLI.

## Local CLI Usage

### Single Image Editing

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png
```

Then run:

```bash
python image_edit.py \
  --image qwen-bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output output_image_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0
```

### Multiple Image Editing (Qwen-Image-Edit-2509)

For multiple image inputs, use `Qwen/Qwen-Image-Edit-2509` or  `Qwen/Qwen-Image-Edit-2511`:

```bash
python image_edit.py \
  --model Qwen/Qwen-Image-Edit-2509 \
  --image img1.png img2.png \
  --prompt "Combine these images into a single scene" \
  --output output_image_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --guidance-scale 1.0
```

### BAGEL Image-To-Image

Use the same generic `image_edit.py` entrypoint for BAGEL. The script detects
the BAGEL pipeline after loading the model and formats the request as
`modalities=["img2img"]` with `multi_modal_data["img2img"]`.

```bash
python image_edit.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --image input.png \
  --prompt "Make the scene look like a watercolor painting" \
  --height 512 \
  --width 512 \
  --num-inference-steps 50 \
  --extra-args '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5}' \
  --negative-prompt "blurry, low quality" \
  --seed 42 \
  --output bagel_img2img.png
```

Key arguments:

- `--model`: model name or path. Use `Qwen/Qwen-Image-Edit-2509` or later for multiple image support.
- `--image`: path(s) to the source image(s) (PNG/JPG, converted to RGB). Can specify multiple images.
- `--prompt` / `--negative-prompt`: text description (string).
- `--extra-args`: JSON object copied to `OmniDiffusionSamplingParams.extra_args` for model-declared controls such as BAGEL's `cfg_text_scale` and `cfg_img_scale`.
- `--cfg-scale`: true classifier-free guidance scale (default: 4.0). Classifier-free guidance is enabled by setting cfg_scale > 1 and providing a negative_prompt. Higher guidance scale encourages images closely linked to the text prompt, usually at the expense of lower image quality.
- `--guidance-scale`: guidance scale for guidance-distilled models (default: 1.0, disabled). Unlike classifier-free guidance (--cfg-scale), guidance-distilled models take the guidance scale directly as an input parameter. Enabled when guidance_scale > 1. Ignored when not using guidance-distilled models.
- `--num-inference-steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--output`: path to save the generated PNG.
- `--vae-use-slicing`: enable VAE slicing for memory optimization.
- `--vae-use-tiling`: enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](../../../docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--strength`: **Z-Image only** - controls the denoising start timestep for I2I (default: 0.6). Range: [0.0, 1.0]. Lower values preserve more of the original image; higher values allow more creative changes.

> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

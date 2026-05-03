# SenseNova-U1-8B-MoT

## Architecture

SenseNova-U1 is a unified Qwen3-based LLM with Mixture-of-Tokenizers (MoT) attention. Unlike two-stage pipelines (e.g., BAGEL), it handles text encoding, optional reasoning ("think mode"), and flow-matching-based image denoising entirely within a single diffusion stage.

| Feature | Description |
| :------ | :---------- |
| **Model type** | Unified LLM (single-stage diffusion pipeline) |
| **Base LLM** | Qwen3 with MoT attention and 3D RoPE |
| **Image generation** | Flow-matching Euler sampler, no separate VAE |
| **Think mode** | Optional chain-of-thought reasoning before image generation |
| **Parallelism** | Tensor Parallelism (TP) with fused QKV and fused gate/up projections |

## Quick Start

```bash
cd examples/offline_inference/sensenova_u1

# Basic text-to-image
python end2end.py --prompt "A cute cat"

# With think mode (model reasons about the prompt first)
python end2end.py --prompt "A cute cat" --think --print-think

# Custom resolution
python end2end.py --prompt "A futuristic cityscape at sunset" \
                  --width 2048 --height 1024 --think
```

> **Note**: Default configuration works on a single **NVIDIA A100 (80GB)** or **H100** GPU.

## Think Mode

When `--think` is enabled, the model generates a chain-of-thought reasoning about how to compose the image before actually generating it. This typically produces higher-quality, more coherent images.

```bash
python end2end.py \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light" \
    --width 1536 --height 2720 \
    --think --print-think
```

Use `--print-think` to display the generated reasoning text.

## Full Parameter Reference

### Generation Parameters

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `--prompt` | "A cute cat..." | Text prompt for image generation |
| `--height` | 2048 | Height of generated image (pixels) |
| `--width` | 2048 | Width of generated image (pixels) |
| `--seed` | 42 | Random seed for reproducibility |
| `--num-steps` | 50 | Number of denoising steps |
| `--cfg-scale` | 4.0 | Classifier-free guidance scale |
| `--cfg-norm` | "none" | CFG normalization mode |
| `--timestep-shift` | 3.0 | Timestep shift for flow-matching schedule |
| `--t-eps` | 0.02 | Epsilon for timestep schedule |
| `--think` | False | Enable think mode |
| `--print-think` | False | Print think text to stdout |

### Infrastructure Parameters

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `--model` | `SenseNova/SenseNova-U1-8B-MoT` | HuggingFace model ID or local path |
| `--deploy-config` | `vllm_omni/deploy/sensenova_u1.yaml` | Deploy YAML configuration |
| `--output` | `.` | Output directory for saved images |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--enforce-eager` | False | Disable torch.compile |

## Reproducing the E2E Test

The following command reproduces the pixel-validated CI test case:

```bash
python end2end.py \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look. The portrait should feel polished and natural, with sharp eyes, realistic skin texture, accurate facial anatomy, and premium lighting that keeps the face as the main focus." \
    --width 1536 --height 2720 \
    --seed 42 --num-steps 50 \
    --cfg-scale 4.0 --timestep-shift 3.0 --cfg-norm none \
    --think --print-think \
    --output outputs
```

The corresponding pytest:

```bash
pytest -s -v tests/e2e/offline_inference/test_sensenova_u1_t2i.py \
    -m "advanced_model" --run-level "advanced_model"
```

# BAGEL-7B-MoT

> BAGEL image generation with vLLM-Omni and OpenAI-compatible serving

## Summary

- Vendor: ByteDance Seed
- Model: `ByteDance-Seed/BAGEL-7B-MoT`
- Task: Text-to-image, image-to-image, and multimodal generation
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When To Use This Recipe

Use this recipe when you want to run BAGEL through the in-repo BAGEL examples
or verify that BAGEL-specific online generation parameters are routed through
`extra_body` into `OmniDiffusionSamplingParams.extra_args`.

## References

- Upstream model: [`ByteDance-Seed/BAGEL-7B-MoT`](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- Offline example: [`examples/offline_inference/bagel/end2end.py`](../../examples/offline_inference/bagel/end2end.py)
- Online client: [`examples/online_serving/bagel/openai_chat_client.py`](../../examples/online_serving/bagel/openai_chat_client.py)
- Server scripts:
  [`examples/online_serving/bagel/run_server.sh`](../../examples/online_serving/bagel/run_server.sh),
  [`examples/online_serving/bagel/run_server_stage_cli.sh`](../../examples/online_serving/bagel/run_server_stage_cli.sh)
- Default deploy configs:
  [`vllm_omni/deploy/bagel.yaml`](../../vllm_omni/deploy/bagel.yaml),
  [`vllm_omni/deploy/bagel_single_stage.yaml`](../../vllm_omni/deploy/bagel_single_stage.yaml)

## Hardware Support

The default BAGEL deploy config places both stages on one 80 GB GPU. For more
headroom, copy the deploy config and move the diffusion stage to another GPU.

### 1x A100 80GB

#### Offline Command

Run the BAGEL offline example from the repository root:

```bash
python examples/offline_inference/bagel/end2end.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --prompt "A beautiful sunset over mountains" \
  --mode text2image \
  --height 512 \
  --width 512 \
  --output /tmp/bagel_text2img.png
```

#### Online Command

Start the OpenAI-compatible server:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091
```

To use the single-stage topology online:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/bagel_single_stage.yaml
```

Send a text-to-image request with BAGEL-specific generation parameters:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "<|im_start|>A beautiful sunset over mountains<|im_end|>"}
        ]
      }
    ],
    "modalities": ["image"],
    "extra_body": {
      "height": 512,
      "width": 512,
      "num_inference_steps": 50,
      "cfg_text_scale": 4.0,
      "cfg_img_scale": 1.5,
      "negative_prompt": "blurry, low quality",
      "seed": 42
    }
  }'
```

The important part is that BAGEL-specific keys such as `cfg_text_scale`,
`cfg_img_scale`, `cfg_interval`, `cfg_renorm_type`, and `cfg_renorm_min` belong
in `extra_body`. The serving layer routes the declared keys into
`OmniDiffusionSamplingParams.extra_args` for the BAGEL pipeline.

#### Verification

Decode the returned data URL into an image:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "<|im_start|>A ceramic teapot on a wooden table<|im_end|>"}
        ]
      }
    ],
    "modalities": ["image"],
    "extra_body": {
      "height": 512,
      "width": 512,
      "num_inference_steps": 25,
      "cfg_text_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[]?.message.content[]? | select(.image_url.url) | .image_url.url' \
    | head -n 1 \
    | cut -d',' -f2- \
    | base64 -d > /tmp/bagel_online.png

ls -lh /tmp/bagel_online.png
```

### 2x CUDA GPUs

Create a custom deploy config from `vllm_omni/deploy/bagel.yaml` and move the
diffusion stage to GPU 1:

```yaml
stages:
  - stage_id: 0
    devices: "0"
    # keep the remaining stage-0 settings from bagel.yaml
  - stage_id: 1
    devices: "1"
    # keep the remaining stage-1 settings from bagel.yaml
```

Then start serving with that config:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT \
  --omni \
  --port 8091 \
  --deploy-config /path/to/custom_bagel_2gpu.yaml
```

Use the online curl request from the `1x A100 80GB` section to verify that the
server returns an image.

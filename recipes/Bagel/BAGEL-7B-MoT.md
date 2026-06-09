# BAGEL-7B-MoT

> BAGEL image generation through the shared online and offline image examples

## Summary

- Vendor: ByteDance Seed
- Model: `ByteDance-Seed/BAGEL-7B-MoT`
- Task: Text-to-image and image-to-image generation
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run BAGEL through the shared image examples
instead of model-specific example clients. The generic examples can format
BAGEL text-to-image and image-to-image prompts, select the image output
modality, attach reference images, and forward BAGEL-specific generation
parameters through the pipeline-declared `extra_args` contract.

## References

- Upstream model:
  [`ByteDance-Seed/BAGEL-7B-MoT`](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- Related offline example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Related offline image-to-image example:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Related online example:
  [`examples/online_serving/text_to_image/openai_chat_client.py`](../../examples/online_serving/text_to_image/openai_chat_client.py)
- Related online image-to-image example:
  [`examples/online_serving/image_to_image/openai_chat_client.py`](../../examples/online_serving/image_to_image/openai_chat_client.py)
- Default deploy configs:
  [`vllm_omni/deploy/bagel.yaml`](../../vllm_omni/deploy/bagel.yaml),
  [`vllm_omni/deploy/bagel_single_stage.yaml`](../../vllm_omni/deploy/bagel_single_stage.yaml)

## Hardware Support

This recipe documents the CUDA layouts used by the in-repo BAGEL deploy
configs. The default two-stage config shares one 80 GB GPU; for more headroom,
move the diffusion stage to a second GPU in a custom deploy config.

## GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA environment with one A100 80 GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Offline Commands

Run text-to-image with the shared offline example from the repository root:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --prompt "A beautiful sunset over mountains" \
  --height 512 \
  --width 512 \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --negative-prompt "blurry, low quality" \
  --seed 42 \
  --output /tmp/bagel_text2img.png
```

Run image-to-image with the shared offline image-to-image example:

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --prompt "Make the scene look like a watercolor painting" \
  --image /path/to/input.jpg \
  --height 512 \
  --width 512 \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --extra-args '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5}' \
  --negative-prompt "blurry, low quality" \
  --seed 42 \
  --output /tmp/bagel_img2img.png
```

To force the single-stage topology:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --stage-configs-path vllm_omni/deploy/bagel_single_stage.yaml \
  --prompt "A beautiful sunset over mountains" \
  --height 512 \
  --width 512 \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --seed 42 \
  --output /tmp/bagel_text2img_single_stage.png
```

The text-to-image example detects BAGEL-style pipelines and wraps prompts as
`<|im_start|>...<|im_end|>`. The image-to-image example keeps the same generic
CLI shape used by other edit models, and only switches to BAGEL's
`modalities: ["img2img"]` / `multi_modal_data["img2img"]` format after the
loaded pipeline is identified as BAGEL.

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

Send a text-to-image request with the OpenAI-compatible image endpoint:

```bash
python examples/online_serving/text_to_image/openai_chat_client.py \
  --server http://localhost:8091 \
  --prompt "A beautiful sunset over mountains" \
  --height 512 \
  --width 512 \
  --steps 50 \
  --cfg-scale 4.0 \
  --negative "blurry, low quality" \
  --seed 42 \
  --output /tmp/bagel_online_text2img.png
```

Send image-to-image with the shared online image-to-image client:

```bash
python examples/online_serving/image_to_image/openai_chat_client.py \
  --server http://localhost:8091 \
  --prompt "Make the scene look like a watercolor painting" \
  --input /path/to/input.jpg \
  --height 512 \
  --width 512 \
  --steps 50 \
  --extra-body '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5}' \
  --negative "blurry, low quality" \
  --seed 42 \
  --output /tmp/bagel_online_img2img.png
```

Equivalent curl request for image-to-image:

```bash
IMAGE_BASE64=$(base64 -w 0 /path/to/input.jpg)

curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg img "$IMAGE_BASE64" '{
    messages: [{
      role: "user",
      content: [
        {type: "text", text: "Make the scene look like a watercolor painting"},
        {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + $img)}}
      ]
    }],
    modalities: ["image"],
    extra_body: {
      height: 512,
      width: 512,
      num_inference_steps: 50,
      cfg_text_scale: 4.0,
      cfg_img_scale: 1.5,
      negative_prompt: "blurry, low quality",
      seed: 42
    }
  }')"
```

The important parts are:

- `modalities: ["image"]` requests image output from the chat endpoint.
- Passing an image in message content triggers the shared serving path to build
  an image-to-image request for pipelines that support it.
- Generation controls belong in `extra_body`; declared BAGEL-specific keys
  such as `cfg_text_scale`, `cfg_img_scale`, `cfg_interval`,
  `cfg_renorm_type`, and `cfg_renorm_min` are routed to
  `OmniDiffusionSamplingParams.extra_args`.

#### Verification

For offline runs, check that the image file was written:

```bash
ls -lh /tmp/bagel_text2img.png
```

For online runs, decode the returned data URL into an image:

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

#### Notes

- Memory usage: the default `bagel.yaml` places both stages on GPU 0 and is
  intended for an 80 GB GPU. For dual-GPU runs, copy the deploy YAML and set
  stage 1 `devices: "1"`.
- Key flags: `--omni` enables vLLM-Omni serving; `--deploy-config` selects a
  custom topology such as `bagel_single_stage.yaml`.
- Offline compatibility: use the generic `text_to_image.py` example for
  text-to-image and the generic `image_edit.py` example for image-to-image.
  Dedicated BAGEL example clients are intentionally not needed.
- This recipe focuses on image generation. BAGEL text-output workflows
  (`img2text`, `text2text`, think-only text) are model capabilities, but they
  are not exposed through a dedicated BAGEL example in this branch.

### 2x CUDA GPUs

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA environment with two compatible GPUs
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

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

#### Verification

Use either shared online client command from the `1x A100 80GB` section and
check that the server returns an image.

#### Notes

- Splitting stages gives the DiT stage more memory headroom and keeps the
  default shared-memory connector on a single node.
- If stages are placed on different nodes, configure the connector section in
  the deploy YAML and start the stage-0 orchestrator before the headless
  stage-1 worker.

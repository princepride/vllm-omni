# MammothModa2-Preview

## Run examples (MammothModa2-Preview)

Download model
```bash
hf download bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

### Text-to-Image (T2I)

Text-to-image now runs through the shared offline image example
(`examples/offline_inference/text_to_image/text_to_image.py`). See the recipe
`recipes/MammothModa2/MammothModa2-Preview.md` for the full command and the
`extra_body` knobs (`text_guidance_scale`, `cfg_range`, `num_inference_steps`).

### Image-to-Text (I2T)

```python
from PIL import Image
from vllm import SamplingParams
from vllm.multimodal.image import convert_image_mode
from vllm_omni import Omni

def main():
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Summarize this image.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    omni = Omni(
        model="./MammothModa2-Preview",
        deploy_config="vllm_omni/deploy/mammoth_moda2_ar.yaml",
    )
    try:
        outputs = list(
            omni.generate(
                [{
                    "prompt": prompt,
                    "multi_modal_data": {"image": convert_image_mode(Image.open("./image.png"), "RGB")},
                    "additional_information": {"omni_task": ["chat"]},
                }],
                [SamplingParams(temperature=0.2, top_p=0.9, top_k=-1, max_tokens=512, seed=42)],
            )
        )
    finally:
        omni.close()

    ro = getattr(outputs[-1], "request_output", outputs[-1])
    print(ro.outputs[0].text.strip())

if __name__ == "__main__":
    main()
```

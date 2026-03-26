# os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    omni = Omni(
        model="tencent/HunyuanImage-3.0-Instruct",
        mode="text-to-image",
    )

    prompt_dict = {
        "prompt": "A beautiful woman with red hair standing in a garden",
        "modalities": ["image"],
    }

    sampling_params = OmniDiffusionSamplingParams(
        guidance_scale=5.0,
        num_inference_steps=50,
        height=1024,
        width=1024,
    )

    omni_outputs = omni.generate(prompts=[prompt_dict], sampling_params_list=[sampling_params])

    req_out = omni_outputs[0].request_output
    images = req_out.images
    if images:
        output_path = "/proj-tango-pvc/users/zhipeng.wang/workspace/vllm-omni/test_output.png"
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        print("No images generated.")


if __name__ == "__main__":
    main()

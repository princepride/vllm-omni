"""End-to-end test for MagiHuman pipeline via vLLM-Omni."""

import os

os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL_PATH = "/proj-tango-pvc/users/zhipeng.wang/workspace/models/daVinci-MagiHuman"


def main():
    print("Initializing MagiHuman pipeline...")
    omni = Omni(
        model=MODEL_PATH,
        init_timeout=1200,
    )

    prompt = "A woman is walking in the park, birds are singing in the background."

    sampling_params = OmniDiffusionSamplingParams(
        height=272,
        width=480,
        num_inference_steps=4,  # minimal steps for testing
        seed=42,
    )

    print(f"Generating with prompt: {prompt}")
    outputs = omni.generate(
        prompts=[prompt],
        sampling_params_list=[sampling_params],
    )

    print(f"Generation complete. Output type: {type(outputs)}")
    if outputs:
        first = outputs[0]
        print(f"First output: {type(first)}")
        if hasattr(first, "request_output") and first.request_output:
            req_out = first.request_output
            print(f"Request output: {type(req_out)}")
            if hasattr(req_out, "custom_output") and req_out.custom_output:
                custom = req_out.custom_output
                if "video" in custom and custom["video"] is not None:
                    print(f"Video shape: {custom['video'].shape}")
                if "audio" in custom and custom["audio"] is not None:
                    print(f"Audio shape: {custom['audio'].shape}")
            if hasattr(req_out, "images") and req_out.images:
                print(f"Images: {len(req_out.images)}")
        print("SUCCESS: MagiHuman pipeline test completed.")
    else:
        print("WARNING: No outputs returned.")


if __name__ == "__main__":
    main()

import argparse
import json
import os

from vllm_omni.inputs.data import OmniPromptType
from vllm_omni.model_executor.stage_input_processors.bagel import (
    GEN_THINK_SYSTEM_PROMPT,
    VLM_THINK_SYSTEM_PROMPT,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="ByteDance-Seed/BAGEL-7B-MoT",
        help="Path to merged model directory.",
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line (preferred).",
    )
    parser.add_argument("--prompt-type", default="text", choices=["text"])

    parser.add_argument(
        "--modality",
        default="text2img",
        choices=["text2img", "img2img", "img2text", "text2text"],
        help="Modality mode to control stage execution.",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to input image for img2img.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory to save images.",
    )

    # OmniLLM init args
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    parser.add_argument("--worker-backend", type=str, default="process", choices=["process", "ray"])
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Path to deploy YAML. If unset, auto-loads vllm_omni/deploy/bagel.yaml based on the HF model_type.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--height", type=int, default=512, help="Output image height in pixels.")
    parser.add_argument("--width", type=int, default=512, help="Output image width in pixels.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Generic guidance scale.")

    parser.add_argument("--cfg-text-scale", type=float, default=4.0, help="Text CFG scale (default: 4.0)")
    parser.add_argument("--cfg-img-scale", type=float, default=1.5, help="Image CFG scale (default: 1.5)")
    parser.add_argument(
        "--negative-prompt", type=str, default=None, help="Negative prompt for CFG (default: empty prompt)"
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="CFG parallel size: 1=batched (single GPU), 2=parallel with 2 branches (text CFG only), 3=parallel (3 GPUs).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
    parser.add_argument(
        "--cfg-interval",
        type=float,
        nargs=2,
        default=None,
        help="CFG interval [start, end] (default: pipeline default)",
    )
    parser.add_argument(
        "--cfg-renorm-type", type=str, default=None, help="CFG renorm type: global, text_channel, channel"
    )
    parser.add_argument("--cfg-renorm-min", type=float, default=None, help="CFG renorm min")
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g. 'fp8').",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Enable thinking mode: AR stage decodes <think>...</think> planning tokens before image generation.",
    )
    parser.add_argument(
        "--max-think-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens for thinking text generation (default: 1000).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=False,
        help="Enable sampling for text generation (default: greedy).",
    )
    parser.add_argument(
        "--text-temperature",
        type=float,
        default=0.3,
        help="Temperature for text generation sampling (default: 0.3).",
    )
    parser.add_argument(
        "--extra-args",
        type=json.loads,
        default=None,
        help=(
            "JSON object with additional model-specific sampling args. Keys "
            "declared by the pipeline flow into OmniDiffusionSamplingParams.extra_args."
        ),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    model_name = args.model
    prompts: list[OmniPromptType] = []
    try:
        if getattr(args, "txt_prompts", None) and args.prompt_type == "text":
            with open(args.txt_prompts, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            prompts = [ln for ln in lines if ln != ""]
            print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")
        else:
            prompts = args.prompts
    except Exception as e:
        print(f"[Error] Failed to load prompts: {e}")
        raise

    if not prompts:
        prompts = ["A cute cat"]
        print(f"[Info] No prompts provided, using default: {prompts}")

    from PIL import Image

    from vllm_omni.diffusion.diffusion_engine import get_extra_body_params
    from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.entrypoints.openai.utils import resolve_diffusion_od_config

    omni_kwargs = vars(args).copy()
    omni_kwargs.pop("model", None)
    deploy_config = args.deploy_config
    if args.think and deploy_config is None:
        deploy_config = "vllm_omni/deploy/bagel_think.yaml"
        print(f"[Info] Think mode enabled, using deploy config: {deploy_config}")
    if deploy_config:
        omni_kwargs["deploy_config"] = deploy_config

    if args.quantization:
        omni_kwargs["quantization_config"] = args.quantization
    omni_kwargs.pop("extra_args", None)

    omni = Omni(model=model_name, **omni_kwargs)
    engine = getattr(omni, "engine", None)
    od_config = resolve_diffusion_od_config(engine)
    declared_extra_body_params = (
        get_extra_body_params(od_config.model_class_name)
        if od_config is not None and getattr(od_config, "model_class_name", None)
        else frozenset()
    )

    formatted_prompts = []
    for p in prompts:
        if args.modality == "img2img":
            if not args.image_path or not os.path.exists(args.image_path):
                raise ValueError(f"img2img requires --image-path pointing to an existing file, got: {args.image_path}")
            loaded_image = Image.open(args.image_path).convert("RGB")
            think_prefix = f"<|im_start|>{GEN_THINK_SYSTEM_PROMPT}<|im_end|>" if args.think else ""
            final_prompt_text = f"{think_prefix}<|fim_middle|><|im_start|>{p}<|im_end|>"
            prompt_dict = {
                "prompt": final_prompt_text,
                "multi_modal_data": {"img2img": loaded_image},
                "modalities": ["img2img"],
            }
            if args.negative_prompt is not None:
                prompt_dict["negative_prompt"] = args.negative_prompt
            formatted_prompts.append(prompt_dict)
        elif args.modality == "img2text":
            if args.image_path:
                loaded_image = Image.open(args.image_path).convert("RGB")
                think_prefix = f"<|im_start|>system\n{VLM_THINK_SYSTEM_PROMPT}<|im_end|>\n" if args.think else ""
                final_prompt_text = (
                    f"{think_prefix}<|im_start|>user\n<|image_pad|>\n{p}<|im_end|>\n<|im_start|>assistant\n"
                )
                prompt_dict = {
                    "prompt": final_prompt_text,
                    "multi_modal_data": {"image": loaded_image},
                    "modalities": ["text"],
                }
                formatted_prompts.append(prompt_dict)
        elif args.modality == "text2text":
            think_prefix = f"<|im_start|>{VLM_THINK_SYSTEM_PROMPT}<|im_end|>" if args.think else ""
            final_prompt_text = f"{think_prefix}<|im_start|>{p}<|im_end|><|im_start|>"
            prompt_dict = {"prompt": final_prompt_text, "modalities": ["text"]}
            formatted_prompts.append(prompt_dict)
        else:
            think_prefix = f"<|im_start|>{GEN_THINK_SYSTEM_PROMPT}<|im_end|>" if args.think else ""
            final_prompt_text = f"{think_prefix}<|im_start|>{p}<|im_end|>"
            prompt_dict = {"prompt": final_prompt_text, "modalities": ["image"]}
            if args.negative_prompt is not None:
                prompt_dict["negative_prompt"] = args.negative_prompt
            formatted_prompts.append(prompt_dict)

    params_list = omni.default_sampling_params_list
    # Bagel exposes 1 sampling param set for single-stage (DiT-only) and
    # 2 for two-stage (Thinker + DiT).  This heuristic may need updating
    # if future pipelines break that 1:1 mapping.
    is_single_stage = len(params_list) == 1

    diffusion_params_idx = 0 if is_single_stage else (1 if len(params_list) > 1 else 0)
    diffusion_params = params_list[diffusion_params_idx]

    if args.modality in ("text2img", "img2img"):
        diffusion_params.height = args.height  # type: ignore
        diffusion_params.width = args.width  # type: ignore
        diffusion_params.num_inference_steps = args.steps  # type: ignore
        if args.guidance_scale is not None:
            diffusion_params.guidance_scale = args.guidance_scale  # type: ignore
        diffusion_params.cfg_parallel_size = args.cfg_parallel_size  # type: ignore
        if args.seed is not None:
            diffusion_params.seed = args.seed  # type: ignore

    declared_user_args = {
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "cfg_interval": tuple(args.cfg_interval) if args.cfg_interval is not None else None,
        "cfg_renorm_type": args.cfg_renorm_type,
        "cfg_renorm_min": args.cfg_renorm_min,
        "negative_prompt": args.negative_prompt,
    }

    needs_text_gen = is_single_stage and (args.think or args.modality in ("text2text", "img2text"))
    if needs_text_gen:
        declared_user_args.update(
            {
                "think": True if args.think else None,
                "max_think_tokens": args.max_think_tokens,
                "do_sample": args.do_sample,
                "text_temperature": args.text_temperature,
            }
        )
    if args.extra_args:
        declared_user_args.update(args.extra_args)
    apply_declared_extra_args(diffusion_params, declared_extra_body_params, declared_user_args)

    omni_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))

    img_idx = 0
    for req_output in omni_outputs:
        # 2-stage think mode: text output from thinker stage
        ro = getattr(req_output, "request_output", None)
        if ro and getattr(ro, "outputs", None):
            txt = "".join(getattr(o, "text", "") or "" for o in ro.outputs)
            if txt:
                if args.think:
                    print(f"[Think]\n{txt}")
                else:
                    print(f"[Output] Text:\n{txt}")

        # Single-stage DiT: text from custom_output
        custom = getattr(req_output, "_custom_output", {}) or {}
        if custom.get("think_text"):
            print(f"[Think]\n{custom['think_text']}")
        if custom.get("text_output"):
            print(f"[Output] Text:\n{custom['text_output']}")

        images = getattr(req_output, "images", None)
        if not images:
            continue

        for j, img in enumerate(images):
            save_path = os.path.join(args.output, f"output_{img_idx}_{j}.png")
            img.save(save_path)
            print(f"[Output] Saved image to {save_path}")
        img_idx += 1


if __name__ == "__main__":
    main()

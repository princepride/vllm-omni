import os

import torch
from transformers import AutoTokenizer

from vllm_omni.model_executor.models.bagel.bagel_core import BagelConfig, add_special_tokens, prepare_prompts
from vllm_omni.model_executor.models.bagel.bagel_dit import Qwen2ImageGenerator, load_bagel_weights
from vllm_omni.model_executor.models.bagel.qwen2_navit import NaiveCache, Qwen2Config, Qwen2TextEncoder, _VaeCfg


def main():
    model_path = "../models/BAGEL-7B-MoT"
    prompt = "A futuristic city skyline at morning, cyberpunk style"

    print("Init models...")
    # Configs
    # cfg_path = os.path.join(model_path, "config.json")
    # with open(cfg_path) as f:
    #     b_cfg = json.load(f)
    llm_cfg_path = os.path.join(model_path, "llm_config.json")
    llm_config = Qwen2Config.from_json_file(llm_cfg_path)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    # Adjust vocab
    tok_len = getattr(tokenizer, "vocab_size", len(tokenizer))
    llm_config.vocab_size = max(int(llm_config.vocab_size), int(tok_len), max(new_token_ids.values()) + 1)

    # Models
    text_encoder = Qwen2TextEncoder(llm_config).cuda().bfloat16()
    bagel_config = BagelConfig(llm_config=llm_config, vae_config=_VaeCfg(), latent_patch_size=2)
    # VAE is now initialized inside image_generator
    image_generator = Qwen2ImageGenerator(llm_config, bagel_config).cuda().bfloat16()

    # Load Weights (Unified)
    load_bagel_weights(model_path, image_generator, text_encoder)

    # Generation
    print("Generating...")
    kv_cache = NaiveCache(llm_config.num_hidden_layers)
    gen_input, _, _ = prepare_prompts(
        curr_kvlens=[0], curr_rope=[0], prompts=[prompt], tokenizer=tokenizer, new_token_ids=new_token_ids
    )
    for k, v in gen_input.items():
        if torch.is_tensor(v):
            gen_input[k] = v.cuda()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        kv_cache = text_encoder.encode_text(past_key_values=kv_cache, **gen_input)

        output = image_generator.generate(past_key_values=kv_cache, new_token_ids=new_token_ids)

    output.output.save("test3_simple_output.png")
    print("Saved test3_simple_output.png")


if __name__ == "__main__":
    main()

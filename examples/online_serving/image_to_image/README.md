# Image-to-Image Online Serving

OpenAI-compatible image editing client supporting Qwen-Image-Edit, BAGEL, and
other image-to-image models.

## BAGEL Image Editing

Pass BAGEL-specific generation parameters through `--extra-body`:

```bash
python openai_chat_client.py \
  --input input.png \
  --prompt "Make the scene look like a watercolor painting" \
  --server http://localhost:8091 \
  --extra-body '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5}'
```

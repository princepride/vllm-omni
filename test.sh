#!/usr/bin/env bash
set -euo pipefail

cd /proj-tango-pvc/users/zhipeng.wang/workspace/vllm-omni
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

STEPS=25
INPUT_IMAGE=${INPUT_IMAGE:-/tmp/vllm_omni_i2i_input.png}
OUT_DIR=${OUT_DIR:-/tmp/vllm_omni_manual_test}
mkdir -p "$OUT_DIR"

python3 - <<PY
from PIL import Image, ImageDraw
path = "${INPUT_IMAGE}"
img = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(img)
draw.rectangle([120, 160, 390, 360], fill=(80, 140, 220))
draw.ellipse([190, 80, 320, 210], fill=(240, 180, 80))
draw.text((150, 420), "vLLM Omni I2I input", fill=(0, 0, 0))
img.save(path)
print(f"Created input image: {path}")
PY

wait_for_server() {
  local url="$1"
  echo "Waiting for server: $url"
  for i in $(seq 1 180); do
    if curl -fsS "$url/v1/models" >/dev/null 2>&1; then
      echo "Server is ready: $url"
      return 0
    fi
    sleep 5
  done
  echo "Server did not become ready: $url" >&2
  return 1
}

stop_server() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping server pid=$pid"
    kill "$pid" || true
    sleep 5
    pkill -P "$pid" || true
  fi
}

echo "===== 1. Qwen-Image offline T2I ====="
python3 examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps "$STEPS" \
  --seed 42 \
  --output "$OUT_DIR/qwen_image_offline_t2i.png"

echo "===== 2. Qwen-Image online T2I ====="
vllm serve Qwen/Qwen-Image --omni --port 8091 > "$OUT_DIR/qwen_image_server.log" 2>&1 &
SERVER_PID=$!
trap 'stop_server "$SERVER_PID"' EXIT
wait_for_server "http://localhost:8091"

python3 examples/online_serving/text_to_image/openai_chat_client.py \
  --server http://localhost:8091 \
  --prompt "a cup of coffee on the table" \
  --height 1024 \
  --width 1024 \
  --steps "$STEPS" \
  --seed 42 \
  --output "$OUT_DIR/qwen_image_online_t2i.png"

stop_server "$SERVER_PID"
trap - EXIT

echo "===== 3. Qwen-Image-Edit offline I2I ====="
python3 examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-Edit \
  --image "$INPUT_IMAGE" \
  --prompt "Convert this image to watercolor style" \
  --num-inference-steps "$STEPS" \
  --cfg-scale 4.0 \
  --guidance-scale 1.0 \
  --seed 42 \
  --output "$OUT_DIR/qwen_image_edit_offline_i2i.png"

echo "===== 4. Qwen-Image-Edit online I2I ====="
vllm serve Qwen/Qwen-Image-Edit --omni --port 8092 > "$OUT_DIR/qwen_image_edit_server.log" 2>&1 &
SERVER_PID=$!
trap 'stop_server "$SERVER_PID"' EXIT
wait_for_server "http://localhost:8092"

python3 examples/online_serving/image_to_image/openai_chat_client.py \
  --server http://localhost:8092 \
  --input "$INPUT_IMAGE" \
  --prompt "Convert this image to watercolor style" \
  --steps "$STEPS" \
  --guidance 1.0 \
  --seed 42 \
  --output "$OUT_DIR/qwen_image_edit_online_i2i.png"

stop_server "$SERVER_PID"
trap - EXIT

echo "===== 5. BAGEL offline T2I ====="
python3 examples/offline_inference/text_to_image/text_to_image.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --prompt "A beautiful sunset over mountains" \
  --height 512 \
  --width 512 \
  --num-inference-steps "$STEPS" \
  --cfg-scale 4.0 \
  --negative-prompt "blurry, low quality" \
  --seed 42 \
  --output "$OUT_DIR/bagel_offline_t2i.png"

echo "===== 6. BAGEL offline I2I ====="
python3 examples/offline_inference/image_to_image/image_edit.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --image "$INPUT_IMAGE" \
  --prompt "Make the scene look like a watercolor painting" \
  --height 512 \
  --width 512 \
  --num-inference-steps "$STEPS" \
  --extra-args '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5}' \
  --negative-prompt "blurry, low quality" \
  --seed 42 \
  --output "$OUT_DIR/bagel_offline_i2i.png"

echo "===== 7. BAGEL online T2I/I2I ====="
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091 > "$OUT_DIR/bagel_server.log" 2>&1 &
SERVER_PID=$!
trap 'stop_server "$SERVER_PID"' EXIT
wait_for_server "http://localhost:8091"

python3 examples/online_serving/text_to_image/openai_chat_client.py \
  --server http://localhost:8091 \
  --prompt "A beautiful sunset over mountains" \
  --height 512 \
  --width 512 \
  --steps "$STEPS" \
  --cfg-scale 4.0 \
  --negative "blurry, low quality" \
  --seed 42 \
  --output "$OUT_DIR/bagel_online_t2i.png"

python3 examples/online_serving/image_to_image/openai_chat_client.py \
  --server http://localhost:8091 \
  --input "$INPUT_IMAGE" \
  --prompt "Make the scene look like a watercolor painting" \
  --height 512 \
  --width 512 \
  --steps "$STEPS" \
  --extra-body '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5}' \
  --negative "blurry, low quality" \
  --seed 42 \
  --output "$OUT_DIR/bagel_online_i2i.png"

stop_server "$SERVER_PID"
trap - EXIT

echo "===== Outputs ====="
ls -lh "$OUT_DIR"
echo "Done. Output dir: $OUT_DIR"
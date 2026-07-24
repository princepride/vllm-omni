#!/usr/bin/env bash
# Manually validate the shared x-to-text example against all supported model families.
# This is intentionally not a pytest test and is not intended for CI.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${REPO_ROOT}/x_to_text_e2e_outputs"
MODEL_FILTER="all"
TASK_FILTER="all"
INPUT_IMAGE="${REPO_ROOT}/image.png"

BAGEL_MODEL="${BAGEL_MODEL:-ByteDance-Seed/BAGEL-7B-MoT}"
HUNYUAN_MODEL="${HUNYUAN_MODEL:-tencent/HunyuanImage-3.0-Instruct}"
MAMMOTH_MODEL="${MAMMOTH_MODEL:-bytedance-research/MammothModa2-Preview}"
HUNYUAN_DEPLOY_CONFIG="${HUNYUAN_DEPLOY_CONFIG:-${SCRIPT_DIR}/hunyuan_image3_ar_2gpu.yaml}"

usage() {
    cat <<'EOF'
Usage:
  bash examples/offline_inference/x_to_text/run_e2e.sh [options]

Options:
  --model all|bagel|hunyuan|mammoth  Models to run (default: all)
  --task all|t2t|i2t                 Text-output tasks to run (default: all)
  --image PATH                       I2T input image (default: repository image.png)
  --output-dir PATH                  Result directory
  -h, --help                         Show this help

Model paths can be overridden with BAGEL_MODEL, HUNYUAN_MODEL, and
MAMMOTH_MODEL. HUNYUAN_DEPLOY_CONFIG overrides the two-GPU Hunyuan AR
config used by this manual script. PYTHON_BIN overrides Python.

Examples:
  # All three models, T2T and I2T, using ./image.png (six runs)
  bash examples/offline_inference/x_to_text/run_e2e.sh

  # Only Hunyuan I2T with a real image
  bash examples/offline_inference/x_to_text/run_e2e.sh \
    --model hunyuan --task i2t --image /path/to/image.jpg

  # Use local checkpoints
  BAGEL_MODEL=/models/BAGEL-7B-MoT \
  HUNYUAN_MODEL=/models/HunyuanImage-3.0-Instruct \
  MAMMOTH_MODEL=/models/MammothModa2-Preview \
    bash examples/offline_inference/x_to_text/run_e2e.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_FILTER="${2:?--model requires a value}"
            shift 2
            ;;
        --task)
            TASK_FILTER="${2:?--task requires a value}"
            shift 2
            ;;
        --image)
            INPUT_IMAGE="${2:?--image requires a path}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?--output-dir requires a path}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MODEL_FILTER" in
    all|bagel|hunyuan|mammoth) ;;
    *) echo "Invalid --model: $MODEL_FILTER" >&2; exit 2 ;;
esac
case "$TASK_FILTER" in
    all|t2t|i2t) ;;
    *) echo "Invalid --task: $TASK_FILTER" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
if [[ "$TASK_FILTER" != "t2t" && ! -f "$INPUT_IMAGE" ]]; then
    echo "Input image does not exist: $INPUT_IMAGE" >&2
    echo "Place image.png at the repository root or pass --image PATH." >&2
    exit 2
fi

PASS=0
FAIL=0

run_case() {
    local family="$1"
    local task="$2"
    local model="$3"
    local deploy_config="${4:-}"
    local output_file="${OUTPUT_DIR}/${family}_${task}.txt"
    local log_file="${OUTPUT_DIR}/${family}_${task}.log"
    local command=(
        "$PYTHON_BIN" "${SCRIPT_DIR}/x_to_text.py"
        --model "$model"
        --prompt "Explain multimodal inference in three concise sentences."
        --output "$output_file"
    )

    if [[ -n "$deploy_config" ]]; then
        command+=(--deploy-config "$deploy_config")
    fi
    if [[ "$task" == "i2t" ]]; then
        command+=(--image "$INPUT_IMAGE")
        command[5]="Describe this image in detail."
    fi

    echo
    echo "================================================================"
    echo "Running ${family} ${task}"
    echo "Model: ${model}"
    echo "Output: ${output_file}"
    echo "================================================================"

    if "${command[@]}" 2>&1 | tee "$log_file"; then
        echo "PASS: ${family} ${task}"
        PASS=$((PASS + 1))
    else
        local exit_code=${PIPESTATUS[0]}
        echo "FAIL: ${family} ${task} (exit ${exit_code})" >&2
        FAIL=$((FAIL + 1))
    fi
}

run_family() {
    local family="$1"
    local model="$2"
    local deploy_config="${3:-}"
    if [[ "$TASK_FILTER" == "all" || "$TASK_FILTER" == "t2t" ]]; then
        run_case "$family" t2t "$model" "$deploy_config"
    fi
    if [[ "$TASK_FILTER" == "all" || "$TASK_FILTER" == "i2t" ]]; then
        run_case "$family" i2t "$model" "$deploy_config"
    fi
}

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "bagel" ]]; then
    run_family bagel "$BAGEL_MODEL"
fi
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "hunyuan" ]]; then
    run_family hunyuan "$HUNYUAN_MODEL" "$HUNYUAN_DEPLOY_CONFIG"
fi
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "mammoth" ]]; then
    run_family mammoth "$MAMMOTH_MODEL"
fi

echo
echo "================================================================"
echo "Manual x-to-text E2E summary: PASS=${PASS}, FAIL=${FAIL}"
echo "Results: ${OUTPUT_DIR}"
echo "================================================================"

[[ "$FAIL" -eq 0 ]]

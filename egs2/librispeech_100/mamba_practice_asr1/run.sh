#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-none}

set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

# Parse module_config option
module_config="['I', 'I', 'I', 'C', 'M']" # module_config는 여기서 정의하면 됨됨
num_blocks="29"
gpu_id=""  # GPU ID를 지정할 변수 (기본값: 빈 문자열, CUDA_VISIBLE_DEVICES 미설정 시 모든 GPU 사용)
remaining_args=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --module_config)
            module_config="$2"
            shift 2
            ;;
        --num_blocks)
            num_blocks="$2"
            shift 2
            ;;
        --gpu_id)
            gpu_id="$2"
            shift 2
            ;;
        *)
            # Store other arguments to pass through
            remaining_args+=("$1")
            shift
            ;;
    esac
done

# GPU ID가 지정된 경우 CUDA_VISIBLE_DEVICES 설정
if [[ -n "$gpu_id" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    echo "Using GPU: $gpu_id (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Warning: No GPU specified. Using all available GPUs."
    echo "To specify a GPU, use: --gpu_id 0 (or 1, 2, etc.)"
fi

# If module_config is provided, update YAML file
if [[ -n "$module_config" ]]; then
    echo "Updating module_config in ${asr_config}..."
    python3 update_module_configs.py \
        --yaml_path "${asr_config}" \
        --module_config "${module_config}" \
        ${num_blocks:+--num_blocks "${num_blocks}"}
    echo "YAML file updated successfully."
fi

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 6 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_asr_model "valid.cer_ctc.ave_5best.pth" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "${remaining_args[@]}"

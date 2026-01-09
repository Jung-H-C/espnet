#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

지금 마카롱 style true여도 scaling x 1.0으로 설정되었음!! 확인필수!


export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-none}

set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

# Parse gpu_id option
gpu_id=""
remaining_args=()

while [[ $# -gt 0 ]]; do
    case $1 in
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
fi

# Restore remaining arguments
set -- "${remaining_args[@]}"

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 6 \
    --nbpe 1000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.cer_ctc.ave_5best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

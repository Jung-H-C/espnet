#!/usr/bin/env bash
# exp 디렉토리의 12개 폴더에서 valid.cer_ctc.ave_5best.pth를 찾아
# 병렬로 inference를 실행하는 스크립트

export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-none}
set -euo pipefail

# 기본 설정
exp_dir="${exp_dir:-exp}"
checkpoint_name="valid.cer_ctc.ave_5best.pth"
max_parallel="${max_parallel:-8}"  # 동시에 실행할 최대 프로세스 수

# run.sh에서 사용하는 기본 인자들
train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other"
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# exp 디렉토리에서 rank*_asr_train_asr_raw_en_bpe1000_sp 패턴의 폴더 찾기
log "Finding experiment directories in ${exp_dir}..."
declare -a exp_dirs=()
while IFS= read -r dir; do
    exp_dirs+=("${dir}")
done < <(find "${exp_dir}" -maxdepth 1 -type d -name "rank*_asr_train_asr_raw_en_bpe1000_sp" | sort -V)

if [ ${#exp_dirs[@]} -eq 0 ]; then
    log "Error: No experiment directories found in ${exp_dir}"
    exit 1
fi

log "Found ${#exp_dirs[@]} experiment directory(ies)"

# 각 폴더에서 checkpoint 파일 찾기
declare -a checkpoint_pairs=()  # (exp_dir, checkpoint_path) 쌍을 저장
for exp_dir_path in "${exp_dirs[@]}"; do
    checkpoint_path="${exp_dir_path}/${checkpoint_name}"
    if [ -f "${checkpoint_path}" ]; then
        checkpoint_pairs+=("${exp_dir_path}|${checkpoint_path}")
        log "Found checkpoint: ${checkpoint_path}"
    else
        log "Warning: checkpoint not found in ${exp_dir_path}: ${checkpoint_path}"
    fi
done

if [ ${#checkpoint_pairs[@]} -eq 0 ]; then
    log "Error: No checkpoint files found."
    exit 1
fi

log "Selected ${#checkpoint_pairs[@]} checkpoint(s) from ${#exp_dirs[@]} experiment directory(ies):"
for pair in "${checkpoint_pairs[@]}"; do
    IFS='|' read -r exp_dir_path checkpoint_path <<< "${pair}"
    echo "  - $(basename "${exp_dir_path}"): ${checkpoint_name}"
done

# 각 checkpoint에 대해 추론 실행하는 함수
run_inference() {
    local pair="$1"
    IFS='|' read -r exp_dir_path checkpoint_path <<< "${pair}"
    local checkpoint_name=$(basename "${checkpoint_path}")
    local exp_dir_name=$(basename "${exp_dir_path}")
    
    log "Starting inference for ${exp_dir_name}/${checkpoint_name}..."
    
    local log_file="${exp_dir_path}/parallel_inference_${checkpoint_name%.pth}.log"
    
    # asr.sh를 stage 12-13만 실행하도록 호출
    ./asr.sh \
        --lang en \
        --ngpu 1 \
        --nj 16 \
        --gpu_inference true \
        --inference_nj 6 \
        --nbpe 1000 \
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
        --inference_asr_model "${checkpoint_name}" \
        --lm_train_text "data/${train_set}/text" \
        --bpe_train_text "data/${train_set}/text" \
        --stage 12 \
        --stop_stage 13 \
        --asr_exp "${exp_dir_path}" \
        > "${log_file}" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log "✓ Completed inference for ${exp_dir_name}/${checkpoint_name}"
    else
        log "✗ Failed inference for ${exp_dir_name}/${checkpoint_name} (exit code: ${exit_code})"
        cat "${log_file}" | tail -20
    fi
    return $exit_code
}

# 병렬 실행을 위해 함수를 export
export -f run_inference
export -f log
export train_set valid_set test_sets asr_config inference_config

# 병렬 실행 (최대 ${max_parallel}개 동시 실행, 나머지는 자동으로 대기)
log "Starting parallel inference with max ${max_parallel} concurrent jobs..."
printf '%s\n' "${checkpoint_pairs[@]}" | xargs -n 1 -P "${max_parallel}" -I {} bash -c 'run_inference "$@"' _ {}

log "All inference jobs completed!"


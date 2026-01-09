#!/usr/bin/env bash
# Train all 70 models from gpu_id_0_config.txt sequentially
# Each model will be trained with its specific module_config and num_blocks
# GPU 0 is used for all training

export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-none}

set -e
# set -u  # Disable to handle potentially empty variables from file parsing
set -o pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Input file
INPUT_FILE="${SCRIPT_DIR}/gpu_id_0_config.txt"

# Check if input file exists
if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: ${INPUT_FILE} not found"
    exit 1
fi

# Parse the input file and train all models
echo "=================================================================================="
echo "Training All Models from gpu_id_0_config.txt Sequentially"
echo "Using GPU 0"
echo "=================================================================================="
echo ""

# Counter for model index
model_idx=0

# Read the file line by line (skip header line)
exec 3< "${INPUT_FILE}"
while IFS=$'\t' read -r module_config block_params num_blocks total_params <&3; do
    # Skip header and empty lines
    if [[ -z "${module_config:-}" ]] || [[ "${module_config}" == "Block" ]] || [[ "${module_config}" =~ ^# ]]; then
        continue
    fi
    
    # Validate required fields
    if [[ -z "${num_blocks:-}" ]]; then
        echo "Warning: Skipping line with empty num_blocks: ${module_config}" >&2
        continue
    fi
    
    # Increment model index
    model_idx=$((model_idx + 1))
    
    # Convert block structure string to list format
    # Example: "BBBC" -> "['B','B','B','C']"
    MODULE_CONFIG_LIST=$(echo "${module_config}" | grep -o . | sed "s/^/'/;s/$/'/" | tr '\n' ',' | sed 's/,$//' | sed "s/^/[/;s/$/]/")
    
    echo "=================================================================================="
    echo "Model ${model_idx}/70: Training"
    echo "=================================================================================="
    echo "Block Structure: ${module_config}"
    echo "Module Config (list format): ${MODULE_CONFIG_LIST}"
    echo "Block Params (M): ${block_params}"
    echo "Number of Blocks: ${num_blocks}"
    echo "Total Params (M): ${total_params}"
    echo "GPU: 0"
    echo ""
    
    # Create unique exp directory for this model (using block structure)
    # Sanitize module_config for directory name (replace special chars)
    SAFE_CONFIG=$(echo "${module_config}" | sed 's/[^A-Za-z0-9]/_/g')
    EXP_DIR="${SCRIPT_DIR}/exp/gpu0_${SAFE_CONFIG}_asr_train_asr_raw_en_bpe1000_sp"
    
    echo "Starting training..."
    echo "Experiment directory: ${EXP_DIR}"
    echo ""
    
    # Run training (disable set -e temporarily to handle errors gracefully)
    set +e
    ./run.sh \
        --gpu_id 0 \
        --module_config "${MODULE_CONFIG_LIST}" \
        --num_blocks "${num_blocks}" \
        --asr_exp "${EXP_DIR}" \
        --stage 11 \
        --stop_stage 11
    TRAIN_EXIT_CODE=$?
    set -e
    
    if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
        echo ""
        echo "✓ Model ${model_idx} (${module_config}) training completed successfully"
        
        # Check if checkpoint exists
        CHECKPOINT_FILE="${EXP_DIR}/valid.cer_ctc.ave_5best.pth"
        
        if [[ -f "${CHECKPOINT_FILE}" ]]; then
            echo "✓ Checkpoint saved: ${CHECKPOINT_FILE}"
        else
            echo "⚠ Warning: Checkpoint not found at ${CHECKPOINT_FILE}"
        fi
        
        # Remove all .pth files except valid.cer_ctc.ave_5best.pth, 10epoch.pth, and 25epoch.pth to save disk space
        if [[ -d "${EXP_DIR}" ]]; then
            DELETED_COUNT=0
            while IFS= read -r -d '' pth_file; do
                # Get filename only for comparison
                filename=$(basename "${pth_file}")
                # Keep valid.cer_ctc.ave_5best.pth, 10epoch.pth, and 25epoch.pth
                if [[ "${pth_file}" != "${CHECKPOINT_FILE}" ]] && \
                   [[ "${filename}" != "10epoch.pth" ]] && \
                   [[ "${filename}" != "25epoch.pth" ]]; then
                    rm -f "${pth_file}"
                    DELETED_COUNT=$((DELETED_COUNT + 1))
                fi
            done < <(find "${EXP_DIR}" -name "*.pth" -type f -print0 2>/dev/null)
            
            if [[ ${DELETED_COUNT} -gt 0 ]]; then
                echo "✓ Removed ${DELETED_COUNT} .pth file(s) (kept ${CHECKPOINT_FILE}, 10epoch.pth, 25epoch.pth)"
            fi
        fi
    else
        echo ""
        echo "✗ Model ${model_idx} (${module_config}) training failed!"
        echo "Continuing with next model..."
    fi
    
    echo ""
    echo "=================================================================================="
    echo ""
    
done
exec 3<&-

echo "=================================================================================="
echo "All 70 models training completed!"
echo "=================================================================================="


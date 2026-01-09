#!/usr/bin/env bash
# Train top 20 models from top80_jacob_cov_sorted_by_synflow.txt sequentially
# Each model will be trained with its specific module_config and num_blocks

export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-none}

set -e
# set -u  # Disable to handle potentially empty variables from file parsing
set -o pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Input file
INPUT_FILE="${SCRIPT_DIR}/top80_jacob_cov_sorted_by_synflow.txt"

# Check if input file exists
if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: ${INPUT_FILE} not found"
    exit 1
fi

# Parse the input file and extract top 20 models
echo "=================================================================================="
echo "Training Top 20 Models Sequentially"
echo "=================================================================================="
echo ""

# Counter for rank
rank=0

# Read the file line by line (skip header lines)
exec 3< "${INPUT_FILE}"
while IFS=$'\t' read -r module_config jacob_cov synflow num_params num_blocks total_params status <&3; do
    # Skip header and empty lines
    if [[ -z "${module_config:-}" ]] || [[ "${module_config}" =~ ^# ]]; then
        continue
    fi
    
    # Validate required fields
    if [[ -z "${num_blocks:-}" ]]; then
        echo "Warning: Skipping line with empty num_blocks: ${module_config}" >&2
        continue
    fi
    
    # Increment rank
    rank=$((rank + 1))
    
    # Only process top 20
    if [[ ${rank} -gt 20 ]]; then
        break
    fi
    
    echo "=================================================================================="
    echo "Rank ${rank}/20: Training model"
    echo "=================================================================================="
    echo "Module Config: ${module_config}"
    echo "Jacob Cov Score: ${jacob_cov}"
    echo "Synflow Score: ${synflow}"
    echo "Number of Blocks: ${num_blocks}"
    echo "Total Params: ${total_params}"
    echo ""
    
    # Run training
    # Create unique exp directory for this model (rank-based)
    EXP_DIR="${SCRIPT_DIR}/exp/rank${rank}_asr_train_asr_raw_en_bpe1000_sp"
    
    echo "Starting training..."
    echo "Experiment directory: ${EXP_DIR}"
    echo ""
    
    # Run training (disable set -e temporarily to handle errors gracefully)
    set +e
    ./run.sh \
        --module_config "${module_config}" \
        --num_blocks "${num_blocks}" \
        --asr_exp "${EXP_DIR}" \
        --stage 10 \
        --stop_stage 11
    TRAIN_EXIT_CODE=$?
    set -e
    
    if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
        echo ""
        echo "✓ Rank ${rank} training completed successfully"
        
        # Check if checkpoint exists
        CHECKPOINT_FILE="${EXP_DIR}/valid.cer_ctc.ave_5best.pth"
        
        if [[ -f "${CHECKPOINT_FILE}" ]]; then
            echo "✓ Checkpoint saved: ${CHECKPOINT_FILE}"
        else
            echo "⚠ Warning: Checkpoint not found at ${CHECKPOINT_FILE}"
        fi
    else
        echo ""
        echo "✗ Rank ${rank} training failed!"
        echo "Continuing with next model..."
    fi
    
    echo ""
    echo "=================================================================================="
    echo ""
    
done
exec 3<&-

echo "=================================================================================="
echo "All 20 models training completed!"
echo "=================================================================================="


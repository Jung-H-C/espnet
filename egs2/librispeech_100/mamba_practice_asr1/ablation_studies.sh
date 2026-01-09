#!/usr/bin/env bash
set -euo pipefail

# # 1) synflow first
# ./run.sh --bpemode bpe \
#          --expdir exp/synflow_first \
#          --module_config "['I', 'RFF', 'RFF', 'C', 'M']" \
#          --num_blocks 11 \
#          --stage 3 --stop_stage 11

# 2) jacob_cov first
./run.sh --bpemode bpe \
         --expdir exp/jacob_cov_first \
         --module_config "['C', 'M', 'C', 'M', 'C']" \
         --num_blocks 13 \
         --stage 12 --stop_stage 13
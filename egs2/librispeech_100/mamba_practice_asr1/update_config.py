#!/usr/bin/env python3
# 모듈별 파라미터 수 (Million)
MODULE_PARAMS = {
    'C': 0.206,  # Conv Module
    'B': 0.482,  # BiMamba Module
    'F': 0.526   # FeedForward Network
}

TARGET_PARAMS = 21.5  # 목표 전체 파라미터 수 (Million)

# 파일 읽기
input_file = '/home/gpuadmin/espnet/egs2/librispeech_100/mamba_practice_asr1/gpu_id_0_config.txt'
output_file = '/home/gpuadmin/espnet/egs2/librispeech_100/mamba_practice_asr1/gpu_id_0_config.txt'

with open(input_file, 'r') as f:
    lines = f.readlines()

# 결과 저장
results = []

for line in lines:
    config = line.strip()
    if not config:  # 빈 줄 건너뛰기
        results.append('')  # 빈 줄 유지
        continue
    
    # 블록 파라미터 계산
    block_params = sum(MODULE_PARAMS.get(char, 0) for char in config)
    
    # 21.5M에 가장 근접한 블록 개수 계산 (반올림)
    num_blocks = round(TARGET_PARAMS / block_params)
    
    # 전체 합산 파라미터 수
    total_params = block_params * num_blocks
    
    # 결과 포맷: config \t block_params \t num_blocks \t total_params
    results.append(f"{config}\t{block_params:.3f}\t{num_blocks}\t{total_params:.3f}")

# 파일 쓰기
with open(output_file, 'w') as f:
    for result in results:
        f.write(result + '\n')

print(f"파일 업데이트 완료: {output_file}")


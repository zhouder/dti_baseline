#!/bin/bash

# =========================
# 配置区
# =========================
CHECK_INTERVAL=30          # 每 30 秒检查一次
THRESHOLD_MB=5120          # 5GB = 5120MB
REQUIRED_COUNT=4          # 连续 4 次 = 2 分钟

CMD="python -m src.train --dataset davis --mode warm --dropout 0.2"

# =========================
# 逻辑区
# =========================
echo "[INFO] Start monitoring GPU 0 used memory..."
echo "[INFO] Condition: GPU 0 used memory < ${THRESHOLD_MB} MB for 2 minutes"

count=0

while true; do
    # 只查询 GPU 0 的已用显存（MB）
    USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1p')

    NOW=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$NOW] GPU 0 used memory: ${USED_MB} MB"

    if [ "$USED_MB" -lt "$THRESHOLD_MB" ]; then
        count=$((count + 1))
        echo "[INFO] Below threshold (${count}/${REQUIRED_COUNT})"
    else
        count=0
        echo "[INFO] Reset counter (GPU 0 busy again)"
    fi

    if [ "$count" -ge "$REQUIRED_COUNT" ]; then
        echo "[INFO] Condition satisfied. Start training on GPU 0!"
        echo "[INFO] Running command:"
        echo "$CMD"

        # 绑定 GPU 0
        export CUDA_VISIBLE_DEVICES=0
        eval $CMD
        break
    fi

    sleep $CHECK_INTERVAL
done

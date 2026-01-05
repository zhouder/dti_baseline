#!/usr/bin/env bash

########################################
# 配置区：按需要改
########################################

# 要监控的 GPU（这里只看 0 号）
GPU_ID=0

# 触发条件：当「已使用显存」小于这个值（MB）就启动任务
USED_MEM_THRESHOLD_MB=2000     # 2GB，可自己改，比如 1500

# 检查间隔（秒）
CHECK_INTERVAL=30

# 你的训练命令（这里用你给的命令）
TRAIN_CMD='
python train_coldcv.py \
  --dataset DAVIS \
  --split-mode cold-protein \
  --cv-folds 5 \
  --overall-train 0.7 \
  --overall-val 0.1 \
  --epochs 100 \
  --batch-size 64 \
  --workers 4 \
  --lr 1e-4 \
  --seed 42 \
  --out-dir /root/lanyun-tmp/mol-runs-2026/DAVIS-cold-protein
'

########################################
# 下面一般不用改
########################################

log() {
  echo "[$(date "+%F %T")] $*"
}

# 检查 nvidia-smi 是否存在
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: 找不到 nvidia-smi，请确认是 NVIDIA 显卡环境。"
  exit 1
fi

log "开始监控 GPU ${GPU_ID} 的显存使用情况..."
log "条件：GPU${GPU_ID} 的『已用显存』 < ${USED_MEM_THRESHOLD_MB} MB 时启动训练"
log "检查间隔：${CHECK_INTERVAL} 秒"

while true; do
  used=$(nvidia-smi --id=${GPU_ID} --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  total=$(nvidia-smi --id=${GPU_ID} --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)

  if [[ -z "$used" || -z "$total" ]]; then
    log "WARN: 无法读取 GPU${GPU_ID} 显存信息，${CHECK_INTERVAL} 秒后重试..."
    sleep "$CHECK_INTERVAL"
    continue
  fi

  log "GPU${GPU_ID}: used=${used}MB / total=${total}MB"

  # 当已用显存 < 阈值时，触发任务
  if (( used < USED_MEM_THRESHOLD_MB )); then
    log "检测到 GPU${GPU_ID} 已用显存 < ${USED_MEM_THRESHOLD_MB}MB，开始启动训练任务。"

    # 只用这张卡
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    log "已设置 CUDA_VISIBLE_DEVICES=${GPU_ID}"
    log "即将执行训练命令..."

    # 用 bash -lc 执行，保证登录环境（如 conda）能加载
    bash -lc "$TRAIN_CMD"

    log "训练命令执行结束，脚本退出。"
    exit 0
  fi

  log "条件未满足，${CHECK_INTERVAL} 秒后重试..."
  sleep "$CHECK_INTERVAL"
done

# -*- coding: utf-8 -*-
"""
ChemBERTa 提取脚本（稳健版）
- 手动加载 RobertaModel + safetensors（绕开 AutoModel.from_pretrained 的 metadata 分支）
- CSV 读取优先 pandas，不可用时自动回退到 csv 模块
- 输出文件名：sha1(smiles)[:24].npz（与训练侧一致）
- 目录结构：{out}/{dataset}/{key[:2]}/{key}.npz（分桶避免单目录过多文件）
"""

from __future__ import annotations
import os
import glob
import csv
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Iterable, Tuple, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaModel
from safetensors.torch import load_file
from tqdm import tqdm


def sha1_24(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


# ---------------- CSV 读取：优先 pandas，失败则回退到 csv ----------------
def read_smiles_pandas(csv_path: str) -> List[str]:
    import pandas as pd  # 按需导入
    df = pd.read_csv(csv_path)
    # 常见列名兼容
    cols = [c.lower() for c in df.columns]
    if "smiles" in cols:
        s = df.columns[cols.index("smiles")]
        smiles = df[s].astype(str).tolist()
    else:
        # 无列名或者非常规列名，退回到第一列
        smiles = df.iloc[:, 0].astype(str).tolist()
    smiles = [x.strip() for x in smiles if isinstance(x, str) and x.strip()]
    return smiles


def read_smiles_csv(csv_path: str) -> List[str]:
    smiles: List[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        header = next(reader, None)
        si = 0
        if header:
            h = [c.strip().lower() for c in header]
            si = h.index("smiles") if "smiles" in h else 0
        for row in reader:
            if not row or len(row) <= si:
                continue
            s = row[si].strip()
            if s:
                smiles.append(s)
    return smiles


def read_smiles(csv_path: str) -> List[str]:
    try:
        return read_smiles_pandas(csv_path)
    except Exception:
        return read_smiles_csv(csv_path)


# ---------------- 模型加载：RobertaModel + safetensors ----------------
def load_chemberta_local(model_dir_or_id: str, offline: bool, device: str, dtype: str):
    """
    - 如果传入的是本地目录：从该目录读取 tokenizer / config / model.safetensors
    - 如果传入的是 HF repo_id：需有本地缓存；offline=True 时不会联网
    """
    if offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    use_dtype = torch.float16 if dtype == "fp16" else torch.float32

    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_dir_or_id, local_files_only=offline)

    # 尝试按“本地目录 + safetensors”加载；若不是本地目录或无 safetensors，再回退到 from_pretrained
    st_path = os.path.join(model_dir_or_id, "model.safetensors")
    cfg_path = os.path.join(model_dir_or_id, "config.json")
    try_safetensors = os.path.isdir(model_dir_or_id) and os.path.exists(st_path) and os.path.exists(cfg_path)

    if try_safetensors:
        cfg = RobertaConfig.from_pretrained(model_dir_or_id, local_files_only=True)
        mdl = RobertaModel(cfg)
        sd = load_file(st_path)
        missing, unexpected = mdl.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        mdl = mdl.to(device=device, dtype=use_dtype).eval()
        return tok, mdl

    # 回退（例如传入的是 repo_id 且本地缓存只有 bin）
    mdl = RobertaModel.from_pretrained(
        model_dir_or_id,
        local_files_only=offline,
        use_safetensors=True,
    ).to(device=device, dtype=use_dtype).eval()
    return tok, mdl


# ---------------- 编码 & 保存 ----------------
@torch.inference_mode()
def encode_batches(
    tok: AutoTokenizer,
    mdl: RobertaModel,
    smiles: List[str],
    batch: int,
    device: str,
    pool: str,
) -> Iterable[Tuple[str, np.ndarray]]:
    for i in range(0, len(smiles), batch):
        batch_smi = smiles[i : i + batch]
        enc = tok(batch_smi, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc).last_hidden_state  # [B,L,d]
        if pool == "cls":
            pooled = out[:, 0, :]
        else:
            pooled = out.mean(dim=1)
        feats = pooled.detach().cpu().to(torch.float32).numpy()  # [B,d]
        for s, v in zip(batch_smi, feats):
            yield s, v


def save_npz(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), feat=arr.astype(np.float32))


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", required=True, help="如 '/root/lanyun-tmp/bindingdb_k5/fold*_*.csv'")
    ap.add_argument("--out", required=True, help="输出根目录，如 /root/lanyun-tmp/cache/chemberta")
    ap.add_argument("--dataset", required=True, help="数据集名（用于输出子目录），如 BindingDB / DAVIS / BioSNAP")
    ap.add_argument("--model_dir", required=True, help="本地模型目录或 HF 模型 ID（推荐本地目录）")
    ap.add_argument("--offline", action="store_true", help="离线模式：不触网，必须有本地缓存/目录")
    ap.add_argument("--device", default="cuda", help="cuda/cpu")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    # 模型
    print(f"[Load] model={args.model_dir} | offline={args.offline} | device={args.device} | dtype={args.dtype}")
    tok, mdl = load_chemberta_local(args.model_dir, args.offline, args.device, args.dtype)
    dim = mdl.config.hidden_size
    print(f"[OK] tokenizer={tok.__class__.__name__}  model={mdl.__class__.__name__}  hidden={dim}")

    # CSV 列表
    csv_files = sorted(glob.glob(args.csv_glob))
    if not csv_files:
        print(f"[WARN] no files matched: {args.csv_glob}")
        return
    print(f"[Files] matched: {len(csv_files)}")

    out_root = Path(args.out) / args.dataset
    t0 = time.time()
    seen: Dict[str, bool] = {}  # 去重

    total_inputs = 0
    total_saved = 0

    for f in csv_files:
        smi = read_smiles(f)
        smi = [s for s in smi if s]  # 清洗
        total_inputs += len(smi)

        # 去重（跨 fold 复用）
        uniq = []
        for s in smi:
            k = sha1_24(s)
            if k not in seen:
                seen[k] = True
                uniq.append(s)

        print(f"[{os.path.basename(f)}] n_smiles={len(smi)}  uniq_new={len(uniq)}")
        if not uniq:
            continue

        for s, vec in tqdm(
            encode_batches(tok, mdl, uniq, args.batch, args.device, args.pool),
            desc=f"encode {os.path.basename(f)}",
        ):
            key = sha1_24(s)
            path = out_root / key[:2] / f"{key}.npz"
            if not path.exists():
                save_npz(path, vec)
                total_saved += 1

    dt = time.time() - t0
    print(f"[DONE] inputs={total_inputs}  saved={total_saved}  uniq_total={len(seen)}  took={dt:.1f}s  out={out_root}")


if __name__ == "__main__":
    main()

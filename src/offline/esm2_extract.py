from __future__ import annotations
import argparse, glob, hashlib, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = AutoTokenizer = AutoModel = None

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:24]

def _dtype_of(s: str):
    s = s.lower()
    if s in ("fp16", "float16", "half"): return torch.float16
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    return torch.float32

def save_npz(out_dir: Path, key: str, arr: np.ndarray):
    out = out_dir / f"{key}.npz"
    np.savez_compressed(out, protein=arr.astype(np.float16))

def embed_one_sequence(seq: str, tok, mdl, device, dtype, max_len: int, chunk_len: int) -> np.ndarray:
    """按残基返回特征 (L, H)，对长序列分块推理后拼接；去掉 BOS/EOS。"""
    if len(seq) == 0:
        H = getattr(mdl.config, "hidden_size", 1280)
        return np.zeros((1, H), dtype=np.float32)

    mdl = mdl.to(device)
    pieces = []
    for i in range(0, len(seq), chunk_len):
        sub = seq[i:i+chunk_len]
        t = tok(sub, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_len)
        with torch.no_grad():
            t = {k: v.to(device) for k, v in t.items()}
            use_amp = (dtype != torch.float32)
            with torch.autocast(device_type=("cuda" if device.startswith("cuda") else "cpu"),
                                enabled=use_amp, dtype=dtype if use_amp else torch.float32):
                out = mdl(**t)
            h = out.last_hidden_state[0, 1:-1, :]  # 去掉 special tokens
            if use_amp and device.startswith("cuda"):
                h = h.to(dtype)
        pieces.append(h.detach().cpu().numpy())
    return np.concatenate(pieces, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", type=str, required=True, help='如 "data/BindingDB/fold*_*.csv"')
    ap.add_argument("--out", type=str, required=True, help="输出根目录（会在下层创建数据集子目录）")
    ap.add_argument("--dataset", type=str, default=None, help="可选：BindingDB / DAVIS / BioSNAP；若给出则写到 out/<dataset>/")
    ap.add_argument("--model_dir", type=str, default="/root/lanyun-tmp/hf/esm2_t33_650M_UR50D",
                    help="本地模型目录（推荐），或 HF 仓库名（需联网）")
    ap.add_argument("--offline", action="store_true", help="只用本地文件，不联网（local_files_only=True）")
    ap.add_argument("--device", type=str, default=("cuda" if torch and torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dtype", type=str, default="fp16", help="fp16|bf16|fp32")
    ap.add_argument("--max_len", type=int, default=1022, help="ESM2 输入最大长度（含 special 约 1024）")
    ap.add_argument("--chunk_len", type=int, default=1000, help="长序列分块长度（不含 special）")
    args = ap.parse_args()

    out = Path(args.out)
    if args.dataset:
        out = out / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.csv_glob))
    if not files:
        print(f"[WARN] No CSV matched: {args.csv_glob}", file=sys.stderr)

    tok = mdl = None
    if AutoTokenizer is None or AutoModel is None:
        print("[INFO] transformers not available; writing zeros as placeholder")
    else:
        local_only = args.offline or os.path.isabs(args.model_dir)
        try:
            # 关键：use_fast=False 兼容只有 vocab.txt 的布局；offline 时不联网
            tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=local_only, use_fast=False)
            mdl = AutoModel.from_pretrained(args.model_dir, local_files_only=local_only)
            mdl.eval()
            print(f"[OK] Loaded model from {args.model_dir} (offline={local_only})", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to load model locally: {e}\n→ Will write zero placeholders.", file=sys.stderr)
            tok = mdl = None

    dtype = _dtype_of(args.dtype)
    device = args.device
    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    seen = set()
    n_ok = n_zero = 0
    for f in files:
        df = pd.read_csv(f)
        for seq in df["protein"].astype(str).unique():
            key = _hash(seq)
            if key in seen: continue
            seen.add(key)
            if tok is None or mdl is None:
                H = 1280  # t33_650M hidden size（占位）
                arr = np.zeros((max(1, len(seq)), H), dtype=np.float32)
                n_zero += 1
            else:
                try:
                    arr = embed_one_sequence(seq, tok, mdl, device, dtype, args.max_len, args.chunk_len)
                    n_ok += 1
                except torch.cuda.OutOfMemoryError:
                    try:
                        arr = embed_one_sequence(seq, tok, mdl, device, dtype, args.max_len, max(256, args.chunk_len // 2))
                        n_ok += 1
                        print(f"[OOM] reduced chunk_len and recovered for key={key}", file=sys.stderr)
                    except Exception as e:
                        H = mdl.config.hidden_size
                        arr = np.zeros((max(1, len(seq)), H), dtype=np.float32)
                        n_zero += 1
                        print(f"[FALLBACK] zero vec for key={key}: {e}", file=sys.stderr)
                except Exception as e:
                    H = mdl.config.hidden_size if mdl is not None else 1280
                    arr = np.zeros((max(1, len(seq)), H), dtype=np.float32)
                    n_zero += 1
                    print(f"[ERR] key={key}: {e} → zero vec", file=sys.stderr)
            save_npz(out, key, arr)
    print(f"[DONE] saved: {n_ok} real, {n_zero} zeros, out={out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


def sha1_24(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sniff_csv_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
    except Exception:
        class _D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return _D()


def resolve_dataset_csv(data_root: Path, dataset: str) -> Path:
    # 优先你描述的：/root/lanyun-fs/{dataset}.csv
    p = data_root / f"{dataset}.csv"
    if p.exists():
        return p
    # 兼容：/root/lanyun-fs/{dataset}/{something}.csv
    d = data_root / dataset
    if d.exists() and d.is_dir():
        cands = list(d.glob("*.csv")) + list(d.glob("*.tsv"))
        if len(cands) == 1:
            return cands[0]
    # 兼容 tsv
    p = data_root / f"{dataset}.tsv"
    if p.exists():
        return p
    raise FileNotFoundError(f"Cannot find dataset csv for '{dataset}' under {data_root}")


def find_col(fieldnames: List[str], candidates: List[str]) -> str:
    low = {c.lower(): c for c in fieldnames}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    # 子串匹配兜底
    for name in candidates:
        nl = name.lower()
        for k, orig in low.items():
            if nl in k:
                return orig
    raise KeyError(f"Cannot find any of {candidates} in header={fieldnames}")


def iter_unique_seqs(csv_path: Path) -> List[Tuple[str, str]]:
    dialect = sniff_csv_dialect(csv_path)
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f, dialect=dialect)
        if dr.fieldnames is None:
            raise RuntimeError(f"No header in {csv_path}")
        seq_col = find_col(dr.fieldnames, ["seq", "sequence"])
        uniq: Dict[str, str] = {}
        for row in dr:
            seq = (row.get(seq_col, "") or "").strip()
            if not seq:
                continue
            pid = sha1_24(seq)
            if pid not in uniq:
                uniq[pid] = seq
    return list(uniq.items())  # [(pid, seq), ...]


@torch.inference_mode()
def encode_full_residue_embeddings(
    model: EsmModel,
    tokenizer,
    seq: str,
    device: torch.device,
    fp16: bool,
    overlap: int = 50,
) -> np.ndarray:
    """
    返回 (L, D) 的残基级特征（去掉 special tokens）。
    对超长序列做滑窗 + overlap 平均融合。
    """
    seq = seq.strip()
    if not seq:
        raise ValueError("Empty sequence")

    # Transformers 的 ESM2 典型 max_position_embeddings=1026（含 special tokens）
    max_pos = int(getattr(model.config, "max_position_embeddings", 1026))
    max_res = max_pos - 2
    if max_res <= 0:
        raise RuntimeError(f"Bad max_position_embeddings={max_pos}")

    L = len(seq)

    def run_chunk(chunk: str) -> np.ndarray:
        toks = tokenizer(
            chunk,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=fp16 and device.type == "cuda",
            dtype=torch.float16,
        ):
            out = model(**toks)
            hs = out.last_hidden_state  # (1, T, D)
        # 去掉 BOS/EOS：序列长度 = len(chunk)
        rep = hs[0, 1:1 + len(chunk), :].float().contiguous().cpu().numpy()
        return rep

    if L <= max_res:
        return run_chunk(seq).astype(np.float32)

    D = int(model.config.hidden_size)
    out_sum = np.zeros((L, D), dtype=np.float32)
    out_cnt = np.zeros((L, 1), dtype=np.float32)

    stride = max(1, max_res - overlap)
    for start in range(0, L, stride):
        end = min(L, start + max_res)
        chunk = seq[start:end]
        rep = run_chunk(chunk)  # (clen, D)
        out_sum[start:end] += rep
        out_cnt[start:end] += 1.0
        if end >= L:
            break

    return (out_sum / np.clip(out_cnt, 1.0, None)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser("Encode protein sequences with HF ESM2 (per-residue).")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--model-dir", type=str, required=True, help="HF local dir of facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save-fp16", action="store_true", help="Save embeddings as float16 (smaller). Default float32.")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = EsmModel.from_pretrained(args.model_dir).to(device).eval()
    if args.fp16 and device.type == "cuda":
        model = model.half()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    for ds in args.datasets:
        csv_path = resolve_dataset_csv(data_root, ds)
        out_dir = out_root / ds / "esm2"
        ensure_dir(out_dir)

        items = iter_unique_seqs(csv_path)  # [(pid, seq)]
        pbar = tqdm(items, desc=f"ESM2-HF[{ds}]", ncols=140)

        for pid, seq in pbar:
            out_path = out_dir / f"{pid}.npz"
            if args.skip_existing and out_path.exists():
                continue
            try:
                rep = encode_full_residue_embeddings(model, tokenizer, seq, device, args.fp16)
                if args.save_fp16:
                    rep = rep.astype(np.float16)
                else:
                    rep = rep.astype(np.float32)
                np.savez_compressed(out_path, x=rep, pid=pid, L=np.int32(rep.shape[0]))
            except Exception as e:
                pbar.write(f"[ESM2-HF] failed pid={pid}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
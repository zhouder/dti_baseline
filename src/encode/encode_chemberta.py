#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel


# -------------------------
# utils
# -------------------------
def sha1_24(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_npy(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.dtype == object:
        raise RuntimeError(f"Refuse to save object array: {path}")
    ensure_dir(path.parent)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def resolve_dataset_csv(data_root: Path, name: str) -> Path:
    # /root/lanyun-fs/{name}.csv or /root/lanyun-fs/{name}/(single csv)
    for ext in [".csv", ".tsv"]:
        for n in [name, name.lower(), name.upper()]:
            p = data_root / f"{n}{ext}"
            if p.exists():
                return p
    d = data_root / name
    if d.exists() and d.is_dir():
        cands = list(d.glob("*.csv")) + list(d.glob("*.tsv"))
        if len(cands) == 1:
            return cands[0]
    raise FileNotFoundError(f"Cannot find dataset csv for {name} under {data_root}")


def guess_delim(path: Path) -> str:
    with path.open("rb") as f:
        head = f.read(4096)
    return "," if head.count(b",") >= head.count(b"\t") else "\t"


def pick_header(headers: List[str], names: Sequence[str]) -> str:
    lower = {h.lower(): h for h in headers}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for n in names:
        nl = n.lower()
        for hl, orig in lower.items():
            if nl in hl:
                return orig
    raise KeyError(f"Cannot find column among {list(names)} in headers={headers}")


def iter_smiles(csv_path: Path) -> Tuple[str, str]:
    """
    Yield (did, smiles) row-by-row.
    did is sha1_24(smiles) so it dedupes naturally via file existence.
    """
    delim = guess_delim(csv_path)
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        try:
            headers = next(reader)
        except StopIteration:
            return

    smi_col = pick_header(headers, ["smile", "smiles"])

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f, delimiter=delim)
        for row in dr:
            smi = (row.get(smi_col, "") or "").strip()
            if not smi:
                continue
            yield sha1_24(smi), smi


# -------------------------
# main
# -------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--datasets", nargs="+", default=["drugbank", "kiba", "davis"])

    ap.add_argument(
        "--model",
        type=str,
        default="/root/lanyun-fs/pretrained/hf/DeepChem/ChemBERTa-77M-MLM",
        help="Local dir of ChemBERTa (or HF repo id).",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    data_root, out_root = Path(args.data_root), Path(args.out_root)
    device = torch.device(args.device)

    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    if args.fp16 and device.type == "cuda":
        model = model.half()

    for ds in args.datasets:
        csv_path = resolve_dataset_csv(data_root, ds)
        out_dir = out_root / ds / "chemberta"
        ensure_dir(out_dir)

        # streaming batches
        batch_smiles: List[str] = []
        batch_ids: List[str] = []

        # tqdm total 不好预估（不读两遍），所以用动态进度条
        pbar = tqdm(desc=f"ChemBERTa[{ds}]", unit="mol")

        def flush_batch():
            if not batch_smiles:
                return
            toks = tokenizer(
                batch_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            hs = out.last_hidden_state  # (B, T, H)
            cls = hs[:, 0, :]          # (B, H)

            for did, vec in zip(batch_ids, cls):
                arr = vec.detach().to("cpu").float().contiguous().numpy().astype(np.float32)
                # 1D: (hidden_size,)
                save_npy(out_dir / f"{did}.npy", arr)

            batch_smiles.clear()
            batch_ids.clear()

        for did, smi in iter_smiles(csv_path):
            out_path = out_dir / f"{did}.npy"
            if args.skip_existing and out_path.exists():
                continue
            batch_smiles.append(smi)
            batch_ids.append(did)

            if len(batch_smiles) >= args.batch:
                flush_batch()
                pbar.update(args.batch)

        # last batch
        if batch_smiles:
            n = len(batch_smiles)
            flush_batch()
            pbar.update(n)

        pbar.close()


if __name__ == "__main__":
    main()
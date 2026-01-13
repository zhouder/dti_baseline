#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# EsmForProteinFolding import compatibility
try:
    from transformers import EsmForProteinFolding
except Exception:
    from transformers.models.esm.modeling_esm import EsmForProteinFolding


def sha1_24(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sniff_delim(path: Path) -> str:
    head = path.read_bytes()[:4096]
    return "," if head.count(b",") >= head.count(b"\t") else "\t"


def resolve_csv(data_root: Path, dataset: str) -> Path:
    for ext in [".csv", ".tsv"]:
        p = data_root / f"{dataset}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {dataset}.csv/tsv under {data_root}")


def pick_col(headers: List[str], candidates: List[str]) -> str:
    low = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    # substring fallback
    for c in candidates:
        cl = c.lower()
        for hl, orig in low.items():
            if cl in hl:
                return orig
    raise KeyError(f"Cannot find columns {candidates} in header={headers}")


def clean_seq(seq: str) -> str:
    """
    ESM tokenizer supports X; replace any non-20AA with X.
    """
    s = seq.strip().upper()
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", s)
    return s


def load_unique_pid_uid_seq(csv_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    pid is defined as sha1_24(original_seq_stripped) to stay consistent with your existing outputs.
    Store cleaned sequence for folding.
    """
    delim = sniff_delim(csv_path)
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter=delim)
        headers = next(r)

    uid_col = pick_col(headers, ["uid"])
    seq_col = pick_col(headers, ["seq", "sequence", "protein"])

    pid2: Dict[str, Tuple[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f, delimiter=delim)
        for row in dr:
            uid = (row.get(uid_col, "") or "").strip()
            raw_seq = (row.get(seq_col, "") or "").strip()
            if not raw_seq:
                continue
            pid = sha1_24(raw_seq)  # keep consistent with previous pid definition
            seq = clean_seq(raw_seq)
            pid2.setdefault(pid, (uid, seq))
    return pid2


def pdb_ok(p: Path) -> bool:
    try:
        if p.stat().st_size < 200:
            return False
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "ATOM" in txt


def split_1024_plus_rest(seq: str, max_len: int) -> List[Tuple[int, int, str]]:
    """
    Split into non-overlapping chunks: [1..max_len], [max_len+1..2*max_len], ..., last remainder.
    """
    L = len(seq)
    out = []
    start = 0
    while start < L:
        end = min(L, start + max_len)
        out.append((start, end, seq[start:end]))
        start = end
    return out


@torch.inference_mode()
def seq_to_pdb_str(model, tokenizer, seq: str, device: torch.device, fp16: bool, num_recycles: int) -> str:
    """
    Robust path:
    - tokenize as batch: tokenizer([seq], ...)
    - avoid infer_pdbs (can produce empty outputs on some versions)
    - fallback add_special_tokens True if False fails
    """
    seq = seq.strip().upper()
    if len(seq) == 0:
        raise ValueError("Empty sequence")

    for add_sp in (False, True):
        inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=add_sp)
        if "input_ids" not in inputs:
            continue
        if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1 or inputs["input_ids"].shape[1] == 0:
            continue
        inputs = {k: v.to(device) for k, v in inputs.items()}

        use_amp = (fp16 and device.type == "cuda")
        with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
            outputs = model(**inputs, num_recycles=num_recycles)

        pdbs = model.output_to_pdb(outputs)
        if isinstance(pdbs, (list, tuple)) and len(pdbs) > 0 and isinstance(pdbs[0], str) and ("ATOM" in pdbs[0]):
            return pdbs[0]

    raise RuntimeError("ESMFold produced no valid PDB (no ATOM).")


def write_pdb_atomic(pdb_str: str, out_pdb: Path) -> None:
    tmp = Path(str(out_pdb) + ".tmp")
    ensure_dir(out_pdb.parent)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(pdb_str)
        if not pdb_str.endswith("\n"):
            f.write("\n")
    os.replace(tmp, out_pdb)


def try_fold_one(
    model, tokenizer, seq: str, out_pdb: Path,
    device: torch.device, fp16: bool, num_recycles: int,
    chunk_sizes: List[int],
) -> Tuple[bool, str]:
    """
    GPU-only: try multiple chunk_size values on OOM.
    """
    for cs in chunk_sizes:
        try:
            if hasattr(model, "set_chunk_size"):
                try:
                    model.set_chunk_size(cs)
                except Exception:
                    pass

            pdb_str = seq_to_pdb_str(model, tokenizer, seq, device=device, fp16=fp16, num_recycles=num_recycles)
            write_pdb_atomic(pdb_str, out_pdb)
            if pdb_ok(out_pdb):
                return True, f"ok(chunk={cs},recycles={num_recycles})"
            return False, f"bad_pdb_output(chunk={cs})"
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                torch.cuda.empty_cache()
                continue
            return False, f"runtime_error(chunk={cs}): {msg[:200]}"
        except Exception as e:
            return False, f"error(chunk={cs}): {str(e)[:200]}"
    return False, "oom_all_chunk_sizes"


def main():
    ap = argparse.ArgumentParser("Fill missing PDB with ESMFold (GPU-only, split long by 1024+rest)")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--esmfold-model-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--num-recycles", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--chunk-sizes", type=str, default="4,2", help="try chunk sizes in order on OOM, e.g. 4,2")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("GPU-only script: please run with --device cuda")

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]
    if not chunk_sizes:
        chunk_sizes = [4, 2]

    # tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.esmfold_model_dir)
    model = EsmForProteinFolding.from_pretrained(args.esmfold_model_dir).to(device).eval()
    if args.fp16:
        model = model.half()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    for ds in args.datasets:
        csv_path = resolve_csv(data_root, ds)
        pid2 = load_unique_pid_uid_seq(csv_path)

        pdb_dir = out_root / ds / "pdb"
        seg_dir = out_root / ds / "pdb_segments"
        ensure_dir(pdb_dir)
        ensure_dir(seg_dir)

        have = set(p.stem for p in pdb_dir.glob("*.pdb"))
        missing = sorted([pid for pid in pid2.keys() if pid not in have])

        log_path = out_root / ds / "esmfold_fail.log"
        ensure_dir(log_path.parent)
        logf = log_path.open("a", encoding="utf-8")

        pbar = tqdm(missing, desc=f"ESMFoldFill[{ds}]", ncols=120)
        ok_cnt, fail_cnt = 0, 0

        for pid in pbar:
            uid, seq = pid2.get(pid, ("", ""))
            if not seq:
                fail_cnt += 1
                logf.write(f"{pid}\t{uid}\t0\tmissing_seq\n")
                continue

            out_pdb = pdb_dir / f"{pid}.pdb"
            if args.skip_existing and out_pdb.exists() and pdb_ok(out_pdb):
                continue

            L = len(seq)
            if L <= args.max_len:
                ok, reason = try_fold_one(
                    model, tokenizer, seq, out_pdb,
                    device=device, fp16=args.fp16, num_recycles=args.num_recycles,
                    chunk_sizes=chunk_sizes
                )
                if ok:
                    ok_cnt += 1
                else:
                    fail_cnt += 1
                    logf.write(f"{pid}\t{uid}\t{L}\t{reason}\n")
                continue

            # long: split into 1024 + rest (non-overlap)
            windows = split_1024_plus_rest(seq, args.max_len)
            cand_paths: List[Path] = []
            for i, (st, ed, sub) in enumerate(windows):
                seg_pdb = seg_dir / f"{pid}_seg{i}_{st+1}-{ed}.pdb"
                if args.skip_existing and seg_pdb.exists() and pdb_ok(seg_pdb):
                    cand_paths.append(seg_pdb)
                    continue

                ok, reason = try_fold_one(
                    model, tokenizer, sub, seg_pdb,
                    device=device, fp16=args.fp16, num_recycles=args.num_recycles,
                    chunk_sizes=chunk_sizes
                )
                if ok:
                    cand_paths.append(seg_pdb)
                else:
                    logf.write(f"{pid}\t{uid}\t{L}\tseg{i}[{st+1}-{ed}] {reason}\n")

            if not cand_paths:
                fail_cnt += 1
                logf.write(f"{pid}\t{uid}\t{L}\tall_segments_failed\n")
                continue

            # choose first successful segment as final representative (or you can choose best by pLDDT later)
            # here: prefer the LONGEST successful segment (usually more informative)
            best = max(cand_paths, key=lambda p: int(p.name.split("_")[-1].split("-")[-1].split(".")[0]))
            tmp = Path(str(out_pdb) + ".tmp")
            tmp.write_bytes(best.read_bytes())
            os.replace(tmp, out_pdb)

            if pdb_ok(out_pdb):
                ok_cnt += 1
            else:
                fail_cnt += 1
                logf.write(f"{pid}\t{uid}\t{L}\tbest_segment_bad_output\n")

        logf.close()
        print(f"[{ds}] missing_start={len(missing)} filled_ok={ok_cnt} fail={fail_cnt}")
        print(f"[{ds}] fail_log: {log_path}")

    print("Done.")


if __name__ == "__main__":
    main()

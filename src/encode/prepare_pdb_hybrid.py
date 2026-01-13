#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
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


def guess_delim(path: Path) -> str:
    head = path.read_bytes()[:4096]
    return "," if head.count(b",") >= head.count(b"\t") else "\t"


def resolve_dataset_csv(data_root: Path, dataset: str) -> Path:
    cands = [
        data_root / f"{dataset}.csv",
        data_root / f"{dataset}.tsv",
        data_root / dataset / f"{dataset}.csv",
        data_root / dataset / f"{dataset}.tsv",
        data_root / dataset / "all.csv",
        data_root / dataset / "all.tsv",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find csv/tsv for {dataset} under {data_root}")


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


def load_uid2uniprot(tsv_path: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    if not tsv_path.exists():
        return mp
    with tsv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            uid = (row.get("uid", "") or "").strip()
            up = (row.get("uniprot", "") or "").strip()
            if uid:
                mp[uid] = up
    return mp


def iter_unique_proteins(csv_path: Path) -> List[Tuple[str, str, str]]:
    """
    Return list of (pid, uid, seq) unique by pid(sha1_24(seq)).
    """
    delim = guess_delim(csv_path)
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter=delim)
        headers = next(r)

    uid_col = pick_col(headers, ["uid"])
    seq_col = pick_col(headers, ["seq", "sequence", "protein"])

    uniq: Dict[str, Tuple[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f, delimiter=delim)
        for row in dr:
            uid = (row.get(uid_col, "") or "").strip()
            seq = (row.get(seq_col, "") or "").strip()
            if not seq:
                continue
            pid = sha1_24(seq)
            if pid not in uniq:
                uniq[pid] = (uid, seq)
    return [(pid, uid, seq) for pid, (uid, seq) in uniq.items()]


def find_first_pdb_url(obj) -> Optional[str]:
    """
    AFDB API fields may evolve. We try:
      - direct keys: pdbUrl / pdb_url
      - search any string containing 'alphafold.ebi.ac.uk/files/' and ending with '.pdb' or '.pdb.gz'
    """
    if isinstance(obj, dict):
        for k in ["pdbUrl", "pdb_url", "pdbURL"]:
            v = obj.get(k)
            if isinstance(v, str) and (v.endswith(".pdb") or v.endswith(".pdb.gz")):
                return v
        # recursive search
        for v in obj.values():
            u = find_first_pdb_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for it in obj:
            u = find_first_pdb_url(it)
            if u:
                return u
    elif isinstance(obj, str):
        if "alphafold.ebi.ac.uk/files/" in obj and (obj.endswith(".pdb") or obj.endswith(".pdb.gz")):
            return obj
    return None


def afdb_get_pdb_url(uniprot: str, timeout: int = 30) -> Optional[str]:
    # Prediction API: https://alphafold.ebi.ac.uk/api/prediction/{uniprot}
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    try:
        js = r.json()
    except Exception:
        return None
    return find_first_pdb_url(js)


def download_file(url: str, out_path: Path, timeout: int = 60) -> bool:
    ensure_dir(out_path.parent)
    tmp = Path(str(out_path) + ".tmp")
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            return False
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, out_path)
    return True


def gunzip_to(in_gz: Path, out_path: Path) -> None:
    tmp = Path(str(out_path) + ".tmp")
    with gzip.open(in_gz, "rb") as fin, tmp.open("wb") as fout:
        fout.write(fin.read())
    os.replace(tmp, out_path)


def pdb_sanity(p: Path) -> bool:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return ("ATOM" in txt) and ("END" in txt or "TER" in txt)


def avg_plddt_from_pdb(p: Path) -> float:
    # AlphaFold/ESMFold often store pLDDT in B-factor column of ATOM records.
    # We'll parse columns 61-66 (1-indexed) => python [60:66]
    vals = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("ATOM") and len(line) >= 66:
            s = line[60:66].strip()
            try:
                vals.append(float(s))
            except Exception:
                pass
    return sum(vals) / len(vals) if vals else -1.0


@torch.inference_mode()
def seq_to_pdb_str(model, tokenizer, seq: str, device: torch.device, fp16: bool) -> str:
    seq = seq.strip()
    if hasattr(model, "infer_pdbs"):
        pdbs = model.infer_pdbs([seq])
        if isinstance(pdbs, (list, tuple)) and pdbs and isinstance(pdbs[0], str):
            return pdbs[0]

    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    autocast_on = (fp16 and device.type == "cuda")
    with torch.autocast(device_type="cuda", enabled=autocast_on, dtype=torch.float16):
        outputs = model(**inputs)

    if hasattr(model, "output_to_pdb"):
        pdbs = model.output_to_pdb(outputs)
        if isinstance(pdbs, (list, tuple)) and pdbs and isinstance(pdbs[0], str):
            return pdbs[0]
        if isinstance(pdbs, str):
            return pdbs

    raise RuntimeError("Cannot convert ESMFold outputs to PDB. Upgrade transformers.")


def esmfold_write_pdb(
    model, tokenizer, seq: str, out_pdb: Path, device: torch.device, fp16: bool, max_tries: int = 3
) -> bool:
    for _ in range(max_tries):
        try:
            pdb_str = seq_to_pdb_str(model, tokenizer, seq, device=device, fp16=fp16)
            tmp = Path(str(out_pdb) + ".tmp")
            ensure_dir(out_pdb.parent)
            with tmp.open("w", encoding="utf-8") as f:
                f.write(pdb_str)
                if not pdb_str.endswith("\n"):
                    f.write("\n")
            os.replace(tmp, out_pdb)
            return pdb_sanity(out_pdb)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                # keep GPU-only: caller may reduce chunk_size externally; here just retry
                continue
            return False
        except Exception:
            return False
    return False


def split_windows(seq: str, max_len: int, overlap: int) -> List[Tuple[int, int, str]]:
    assert 0 <= overlap < max_len
    step = max_len - overlap
    L = len(seq)
    windows = []
    start = 0
    idx = 0
    while start < L:
        end = min(L, start + max_len)
        windows.append((start, end, seq[start:end]))
        if end == L:
            break
        start += step
        idx += 1
    return windows


def main():
    ap = argparse.ArgumentParser("Hybrid PDB cache: AFDB download (if UniProt) else ESMFold (GPU)")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-fs")

    ap.add_argument("--davis-map", type=str, default="/root/lanyun-fs/davis/uid2uniprot.tsv",
                    help="TSV with columns uid, uniprot for davis gene->UniProt mapping")

    ap.add_argument("--esmfold-model-dir", type=str, required=True,
                    help="Local HF dir of facebook/esmfold_v1")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=64)

    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--split-long", action="store_true",
                    help="If seq > max-len, fold as overlapping windows on GPU and pick best by avg pLDDT.")

    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.05, help="sleep between AFDB API calls")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    davis_uid2up = load_uid2uniprot(Path(args.davis_map))

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.esmfold_model_dir)
    model = EsmForProteinFolding.from_pretrained(args.esmfold_model_dir).to(device).eval()
    if args.fp16 and device.type == "cuda":
        model = model.half()
    if hasattr(model, "set_chunk_size"):
        try:
            model.set_chunk_size(args.chunk_size)
        except Exception:
            pass

    for ds in args.datasets:
        csv_path = resolve_dataset_csv(data_root, ds)
        pdb_dir = out_root / ds / "pdb"
        seg_dir = out_root / ds / "pdb_segments"
        ensure_dir(pdb_dir)
        ensure_dir(seg_dir)

        items = iter_unique_proteins(csv_path)
        pbar = tqdm(items, desc=f"PreparePDB[{ds}]", ncols=120)

        n_af_ok = n_af_fail = n_esm_ok = n_esm_fail = 0

        for pid, uid, seq in pbar:
            out_pdb = pdb_dir / f"{pid}.pdb"
            if args.skip_existing and out_pdb.exists() and pdb_sanity(out_pdb):
                continue

            # Determine UniProt accession
            uniprot = ""
            if ds.lower() == "davis":
                uniprot = (davis_uid2up.get(uid, "") or "").strip()
            else:
                # kiba/drugbank uid is UniProt (you confirmed)
                uniprot = uid.strip()

            # 1) Try AFDB download if we have UniProt
            downloaded = False
            if uniprot:
                try:
                    pdb_url = afdb_get_pdb_url(uniprot)
                    time.sleep(args.sleep)
                    if pdb_url:
                        # handle .gz
                        if pdb_url.endswith(".pdb.gz"):
                            gz_path = Path(str(out_pdb) + ".gz")
                            ok = download_file(pdb_url, gz_path)
                            if ok:
                                gunzip_to(gz_path, out_pdb)
                                try:
                                    gz_path.unlink()
                                except Exception:
                                    pass
                                downloaded = pdb_sanity(out_pdb)
                        else:
                            ok = download_file(pdb_url, out_pdb)
                            downloaded = ok and pdb_sanity(out_pdb)
                except Exception:
                    downloaded = False

                if downloaded:
                    n_af_ok += 1
                    continue
                else:
                    n_af_fail += 1

            # 2) Fallback to ESMFold (GPU)
            s = seq.strip()
            if (len(s) > args.max_len) and args.split_long:
                windows = split_windows(s, args.max_len, args.overlap)
                cand_paths = []
                for i, (st, ed, sub) in enumerate(windows):
                    cand = seg_dir / f"{pid}_seg{i}_{st+1}-{ed}.pdb"
                    if (not args.skip_existing) or (not cand.exists()) or (not pdb_sanity(cand)):
                        ok = esmfold_write_pdb(model, tokenizer, sub, cand, device=device, fp16=args.fp16)
                        if not ok:
                            continue
                    if pdb_sanity(cand):
                        cand_paths.append(cand)

                if not cand_paths:
                    n_esm_fail += 1
                    pbar.write(f"[ESMFold] failed all segments pid={pid} L={len(s)}")
                    continue

                # pick best segment by avg pLDDT
                best = max(cand_paths, key=avg_plddt_from_pdb)
                tmp = Path(str(out_pdb) + ".tmp")
                tmp.write_bytes(best.read_bytes())
                os.replace(tmp, out_pdb)
                if pdb_sanity(out_pdb):
                    n_esm_ok += 1
                else:
                    n_esm_fail += 1
            else:
                # single fold (if longer than max_len and not split_long, we skip)
                if len(s) > args.max_len and not args.split_long:
                    n_esm_fail += 1
                    pbar.write(f"[ESMFold] skip long pid={pid} L={len(s)} (> {args.max_len}); use --split-long")
                    continue
                ok = esmfold_write_pdb(model, tokenizer, s, out_pdb, device=device, fp16=args.fp16)
                if ok:
                    n_esm_ok += 1
                else:
                    n_esm_fail += 1

        print(f"[{ds}] AFDB ok={n_af_ok} fail={n_af_fail} | ESMFold ok={n_esm_ok} fail={n_esm_fail}")
        print(f"[{ds}] pdb_dir: {pdb_dir}")

    print("Done.")


if __name__ == "__main__":
    main()

import os
import faiss
import numpy as np


def build_index(texts, embed_fn, index_path):
    vecs = embed_fn(texts)
    dim  = vecs.shape[1]
    cpu_idx = faiss.IndexFlatIP(dim)
    cpu_idx.add(vecs)
    faiss.write_index(cpu_idx, index_path)
    print(f"Saved FAISS index in {index_path}  ({cpu_idx.ntotal:,} docs, dim={dim})")
    return _to_gpu(cpu_idx)


def load_index(index_path):
    print(f"Loading FAISS index from '{index_path}'")
    cpu_idx = faiss.read_index(index_path)
    print(f"  {cpu_idx.ntotal:,} docs loaded.")
    return _to_gpu(cpu_idx)


def build_or_load_index(texts, embed_fn, index_path):
    if os.path.exists(index_path):
        print(f"Found cached index at '{index_path}'")
        return load_index(index_path)
    print("No cached index — building from corpus …")
    return build_index(texts, embed_fn, index_path)


def rebuild_index(texts, embed_fn, index_path):
    return build_index(texts, embed_fn, index_path)


def retrieve(index, query_vecs, k,):
    scores, indices = index.search(query_vecs, k)
    return scores, indices


def retrieve_threshold(index, query_vecs, tau, k_max):
    scores_all, indices_all = index.search(query_vecs, k_max)  # [B, k_max]
    results = []
    for s_row, i_row in zip(scores_all, indices_all):
        mask = s_row >= tau
        if mask.any():
            results.append((s_row[mask].tolist(), i_row[mask].tolist()))
        else:
            # always return at least one document
            results.append(([float(s_row[0])], [int(i_row[0])]))
    return results


def _to_gpu(cpu_idx):
    try:
        res = faiss.StandardGpuResources()
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        print("  FAISS index on GPU.")
        return gpu_idx
    except Exception:
        print("  FAISS-GPU not available — using CPU index.")
        return cpu_idx

import sys, os, json, struct, tempfile
from collections import defaultdict
import torch
from safetensors.torch import safe_open


DEFAULT_MODEL_DIR = "../Mistral-7B-v0.1/"
DEFAULT_OUT_PATH  = "../model.bin"

"""
Build a single model file from Hugging Face.

Layout:
  [8-byte little-endian uint64: JSON header size]
  [JSON header bytes]
  [raw tensor bytes, concatenated]
"""

model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
out_path  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_PATH

cfg_path       = os.path.join(model_dir, "config.json")
idx_path       = os.path.join(model_dir, "model.safetensors.index.json")
tokenizer_path = os.path.join(model_dir, "tokenizer.json")

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
with open(idx_path, "r", encoding="utf-8") as f:
    st_idx = json.load(f)

header = {
    "__metadata__": {
        "hidden_dim": str(cfg["intermediate_size"]),
        "n_layers": str(cfg["num_hidden_layers"]),
        "n_kv_heads": str(cfg["num_key_value_heads"]),
        "vocab_size": str(cfg["vocab_size"]),
        "max_seq_len": str(cfg["max_position_embeddings"]),
        "rope_theta": str(cfg["rope_theta"]),
        "norm_eps": str(cfg["rms_norm_eps"]),
        "act_type": cfg["hidden_act"],
        "dtype": "fp32",
    }
}

def build_vocab_blob(tokenizer_json_path: str, vocab_size: int) -> bytes:
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tok = json.load(f)
    vocab = tok["model"]["vocab"]  # token -> id
    words = [b"" for _ in range(vocab_size)]
    for token, idx in vocab.items():
        w = token.replace("\u2581", " ").replace("\0", "\7") + "\0"
        words[idx] = w.encode("utf-8")
    return b"".join(words)

weight_map = st_idx["weight_map"]
by_file = defaultdict(list)
for k, fname in weight_map.items():
    by_file[fname].append(k)

tmp = tempfile.NamedTemporaryFile(prefix="yalm_", delete=False)
tmp_path = tmp.name
offset = 0


vocab_size = int(header["__metadata__"]["vocab_size"])
vocab_blob = build_vocab_blob(tokenizer_path, vocab_size)
start, end = offset, offset + len(vocab_blob)
header["vocab"] = {
    "dtype": "U8",
    "shape": [len(vocab_blob)],  # flat byte buffer
    "data_offsets": [start, end],
}
tmp.write(vocab_blob)
offset = end

# --- Dump tensors as FP32 ---
for fname, keys in by_file.items():
    st_path = os.path.join(model_dir, fname)
    with safe_open(st_path, framework="pt") as sf:
        for key in keys:
            t = sf.get_tensor(key)
            if t.dtype is not torch.float32:
                t = t.to(torch.float32)
            blob = t.contiguous().numpy().tobytes(order="C")
            start, end = offset, offset + len(blob)
            header[key] = {
                "dtype": "F32",
                "shape": list(t.shape)[:4],
                "data_offsets": [start, end],
            }
            tmp.write(blob)
            offset = end

tmp.flush()
tmp.close()

json_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(out_path, "wb") as fout:
    fout.write(struct.pack("<Q", len(json_bytes)))
    fout.write(json_bytes)
    with open(tmp_path, "rb") as fin:
        while (chunk := fin.read(1 << 20)):
            fout.write(chunk)

os.remove(tmp_path)
print(f"OK: wrote {out_path} (header={len(json_bytes)} bytes, payload={offset} bytes)")
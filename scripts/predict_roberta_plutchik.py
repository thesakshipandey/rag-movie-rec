# main/projects/rag-movie-rec/predict_roberta_plutchik.py
#!/usr/bin/env python
import os, json, argparse, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_calibration(path):
    try:
        with open(path, "r") as f:
            return json.load(f).get("temperature", 1.0)
    except Exception:
        return 1.0

def _resolve_device(dev_arg: str) -> torch.device:
    # Accept: auto | cpu | cuda | cuda:0 | cuda:1 | ...
    if dev_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        d = torch.device(dev_arg)
    except Exception:
        d = torch.device("cpu")
    # Pin process to that CUDA device if applicable
    if d.type == "cuda":
        try:
            torch.cuda.set_device(d)
        except Exception:
            pass
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)   # .../final or checkpoint-*
    ap.add_argument("--text", type=str, nargs="+", required=True)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--format", choices=["text","json"], default="text")
    ap.add_argument("--device", type=str, default="auto",
                   help="cpu | cuda | cuda:0 | cuda:1 | auto")
    ap.add_argument("--dtype", type=str, default="float16",
                   choices=["float32","float16","bfloat16"],
                   help="Model inference dtype")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # Make CUDA numbering match nvidia-smi if user exports it
    # (no-op if not set)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    tok = AutoTokenizer.from_pretrained(args.model_dir)

    # Critical: prevent auto device_map; load in chosen dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        device_map=None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    with open(f"{args.model_dir}/label_map.json","r") as f:
        lm = json.load(f)
    id2label = {int(k): v for k, v in lm["id2label"].items()}
    T = load_calibration(f"{args.model_dir}/calibration.json")

    batch = tok(
        args.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=args.max_len
    ).to(device)

    with torch.inference_mode():
        logits = model(**batch).logits / T
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    for q, p in zip(args.text, probs):
        order = p.argsort()[::-1]
        top_idx = int(order[0])
        top_label = id2label[top_idx]
        top_k = min(args.topk, len(p))
        sorted_probs = [(id2label[int(j)], float(round(p[int(j)], 4))) for j in order[:top_k]]
        lines = [f"  - {id2label[int(j)]}: {p[int(j)]:.6f}" for j in order[:top_k]]

        if args.format == "json":
            print(json.dumps({
                "text": q,
                "top": {"label": top_label, "prob": float(round(p[top_idx],4))},
                "probs_sorted": sorted_probs
            }, ensure_ascii=False))
        else:
            print("\nTEXT:", q)
            print("TOP:", f"{id2label[int(order[0])]} ({p[int(order[0])]:.6f})")
            print("PROBS:")
            print("\n".join(lines))

if __name__ == "__main__":
    main()

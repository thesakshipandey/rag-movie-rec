#!/usr/bin/env python
import os, json, math, argparse, numpy as np, torch, random
from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from torch import nn

EMOTIONS = ["Joy","Trust","Fear","Anticipation","Sadness","Anger","Surprise","Disgust"]
label2id = {e:i for i,e in enumerate(EMOTIONS)}
id2label = {i:e for i,e in enumerate(EMOTIONS)}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    X, y = [], []
    for emo, arr in js.items():
        if emo not in label2id: continue
        for t in arr:
            if not isinstance(t, str): continue
            s = t.strip()
            if s:
                X.append(s); y.append(label2id[emo])
    return X, y

def stratify_split(X, y, seed=42):
    from sklearn.model_selection import train_test_split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

@dataclass
class DS:
    encodings: dict
    labels: list
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def encode_texts(tok, texts, max_len):
    return tok(texts, truncation=True, padding=True, max_length=max_len)

# --- Optional: Distillation 28→8 (GoEmotions teacher) ---
def build_ge28_to_plutchik8_map(teacher_id2label: Dict[int,str]) -> Dict[int, List[int]]:
    name2id = {teacher_id2label[i].lower(): i for i in teacher_id2label}
    def ids(*names):
        out = []
        for n in names:
            n = n.lower()
            if n in name2id: out.append(name2id[n])
        return out

    mapping = {
        0: ids("joy","amusement","gratitude","relief","pride","approval","love","optimism","excitement","admiration"),
        1: ids("admiration","approval","caring","gratitude","love","pride","optimism","realization"),
        2: ids("fear","nervousness","embarrassment"),
        3: ids("curiosity","desire","optimism","excitement","realization"),
        4: ids("sadness","grief","disappointment","remorse"),
        5: ids("anger","annoyance","disapproval"),
        6: ids("surprise","realization","amusement","excitement","confusion"),
        7: ids("disgust","disapproval")
    }
    return mapping

class DistillTrainer(Trainer):
    def __init__(self, *args, teacher=None, ge_map=None, kd_alpha=0.3, kd_temp=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.ge_map = ge_map
        self.kd_alpha = kd_alpha
        self.kd_temp = kd_temp

        # Move student & teacher to the trainer device right away
        dev = self.args.device
        base_student = self.model.module if hasattr(self.model, "module") else self.model
        base_student.to(dev)

        if self.teacher is not None:
            base_teacher = self.teacher.module if hasattr(self.teacher, "module") else self.teacher
            for p in base_teacher.parameters():
                p.requires_grad = False
            base_teacher.to(dev)
            base_teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure per-batch device sync (covers any accelerator / DP edge cases)
        dev = None
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                dev = v.device; break
        if dev is None:
            dev = self.args.device

        base_student = model.module if hasattr(model, "module") else model
        if next(base_student.parameters()).device != dev:
            base_student.to(dev)

        if self.teacher is not None:
            base_teacher = self.teacher.module if hasattr(self.teacher, "module") else self.teacher
            if next(base_teacher.parameters()).device != dev:
                base_teacher.to(dev)
                base_teacher.eval()

        labels = inputs.get("labels")

        # Student forward
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        ce = nn.CrossEntropyLoss()(logits, labels)

        if self.teacher is None or self.ge_map is None or self.kd_alpha <= 0:
            return (ce, outputs) if return_outputs else ce

        # Teacher forward on same device
        with torch.no_grad():
            t_outputs = self.teacher(**{k: v for k, v in inputs.items() if k != "labels"})
            t_probs28 = torch.softmax(t_outputs.logits / self.kd_temp, dim=-1)

            B = t_probs28.size(0)
            t_probs8 = torch.zeros((B, 8), device=t_probs28.device)
            for tgt, src_ids in self.ge_map.items():
                if len(src_ids) > 0:
                    t_probs8[:, tgt] = t_probs28[:, src_ids].sum(dim=-1)
            t_probs8 = t_probs8 / (t_probs8.sum(dim=-1, keepdim=True) + 1e-12)

        # KD + CE
        s_log_probs = torch.log_softmax(logits / self.kd_temp, dim=-1)
        kl = torch.sum(t_probs8 * (torch.log(t_probs8 + 1e-12) - s_log_probs), dim=-1).mean()
        kd_loss = (self.kd_temp ** 2) * kl
        loss = (1 - self.kd_alpha) * ce + self.kd_alpha * kd_loss
        return (loss, outputs) if return_outputs else loss

def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    def nll_at_T(T):
        probs = torch.softmax(torch.tensor(logits)/T, dim=-1).numpy()
        eps = 1e-12
        return -np.mean(np.log(probs[np.arange(len(labels)), labels] + eps))
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.7, 5.0, 44):
        nll = nll_at_T(T)
        if nll < best_nll: best_nll, best_T = nll, float(T)
    return float(best_T)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="prompts.json")
    parser.add_argument("--out_dir", type=str, default="roberta-plutchik-query")
    parser.add_argument("--model_name", type=str, default="SamLowe/roberta-base-go_emotions")
    parser.add_argument("--epochs_stage1", type=int, default=1)
    parser.add_argument("--epochs_stage2", type=int, default=3)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=5e-5)
    parser.add_argument("--batch_train", type=int, default=16)
    parser.add_argument("--batch_eval", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--kd_alpha", type=float, default=0.3)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    X, y = load_prompts(args.data_path)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = stratify_split(X, y, seed=args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    enc_tr = encode_texts(tok, X_tr, args.max_len)
    enc_va = encode_texts(tok, X_va, args.max_len)
    enc_te = encode_texts(tok, X_te, args.max_len)
    ds_tr = DS(enc_tr, y_tr); ds_va = DS(enc_va, y_va); ds_te = DS(enc_te, y_te)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(EMOTIONS), ignore_mismatched_sizes=True
    )
    model.config.id2label = {i:e for i,e in enumerate(EMOTIONS)}
    model.config.label2id = {e:i for i,e in enumerate(EMOTIONS)}
    model.config.problem_type = "single_label_classification"

    teacher = None; ge_map = None
    if args.distill:
        teacher = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        if hasattr(teacher.config, "id2label"):
            ge_map = build_ge28_to_plutchik8_map(teacher.config.id2label)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {"macro_f1": macro_f1, "accuracy": acc}

    # Stage 1
    for name, param in model.named_parameters():
        if "classifier" in name: param.requires_grad = True
        else: param.requires_grad = False

    args1 = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "stage1"),
        learning_rate=args.lr_head,
        per_device_train_batch_size=args.batch_train,
        per_device_eval_batch_size=args.batch_eval,
        num_train_epochs=max(1, args.epochs_stage1),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none",
        seed=args.seed
    )

    TrainerClass = DistillTrainer
    extra = {}
    if args.distill:
        extra = dict(teacher=teacher, ge_map=ge_map, kd_alpha=args.kd_alpha, kd_temp=args.kd_temp)

    tr1 = TrainerClass(
        model=model, args=args1, train_dataset=ds_tr, eval_dataset=ds_va,
        processing_class=tok, compute_metrics=compute_metrics, **extra
    )
    tr1.train()

    # Stage 2
    for p in model.parameters(): p.requires_grad = True

    head_params, enc_params = [], []
    for n,p in model.named_parameters():
        (head_params if "classifier" in n else enc_params).append(p)
    optim_groups = [
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ]

    args2 = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "stage2"),
        per_device_train_batch_size=args.batch_train,
        per_device_eval_batch_size=args.batch_eval,
        num_train_epochs=max(1, args.epochs_stage2),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none",
        seed=args.seed
    )

    tr2 = TrainerClass(
        model=model, args=args2, train_dataset=ds_tr, eval_dataset=ds_va,
        processing_class=tok, compute_metrics=compute_metrics,
        optimizers=(torch.optim.AdamW(optim_groups), None),
        teacher=teacher, ge_map=ge_map, kd_alpha=args.kd_alpha, kd_temp=args.kd_temp
    )
    tr2.train()

    print("Validation:", tr2.evaluate(ds_va))
    print("Test:", tr2.evaluate(ds_te))

    preds = tr2.predict(ds_te)
    y_pred = preds.predictions.argmax(-1)
    print("\nTest classification report:\n", classification_report(y_te, y_pred, target_names=EMOTIONS))

    calib = {"temperature": 1.0}
    if args.calibrate:
        val_logits = tr2.predict(ds_va).predictions
        T = temperature_scale(val_logits, np.array(y_va))
        calib["temperature"] = float(T)
        print(f"Calibrated temperature: {T:.3f}")

    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    tr2.save_model(final_dir)
    tok.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "label_map.json"), "w") as f:
        json.dump({"id2label": {i:e for i,e in enumerate(EMOTIONS)},
                   "label2id": {e:i for i,e in enumerate(EMOTIONS)}}, f, indent=2)
    with open(os.path.join(final_dir, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSaved model to: {final_dir}")

if __name__ == "__main__":
    main()

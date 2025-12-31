# -*- coding: utf-8 -*-
import os
import math
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
)

test_csv = "test_data.csv"

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
local_dir = "qwen25_coder_1_5b_local"
save_local = True

seed = 42
max_ctx = 8192

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print("device:", dev)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

df = pd.read_csv(test_csv)
if "function" not in df.columns or "label" not in df.columns:
    raise ValueError("test_csv must have columns: function, label")

df["function"] = df["function"].astype(str)
df["label"] = df["label"].astype(int)

sys_text = """You are a security auditor.

Task:
Given a single function, output:
1 = vulnerable
0 = not vulnerable

General principle:
Output 1 only if the function contains a plausible security weakness where an attacker could cause an unwanted security impact due to missing or insufficient safeguards.
Output 0 otherwise.

How to decide (high-level):
- Consider whether the function processes or depends on input that could be influenced by an attacker.
- Consider whether the function performs any operation that could matter for security (e.g., decisions, access, execution, storage, communication, or exposure of data/resources).
- If attacker influence + insufficient safeguards could realistically lead to harm, output 1.
- If protections are adequate or there is not enough evidence of a real security risk, output 0.

Output constraints (STRICT):
Return exactly one character: 0 or 1.
No explanation. No extra text. No punctuation. No extra whitespace.
"""

load_path = local_dir if os.path.isdir(local_dir) else model_id
tok = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    load_path,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
mdl.to(dev)
mdl.eval()
torch.set_grad_enabled(False)

if save_local and (not os.path.isdir(local_dir)):
    os.makedirs(local_dir, exist_ok=True)
    tok.save_pretrained(local_dir)
    mdl.save_pretrained(local_dir)

t0 = tok("0", add_special_tokens=False).input_ids
t1 = tok("1", add_special_tokens=False).input_ids

def prompt_text(code: str) -> str:
    user = "Now classify the following function:\n<code>\n" + code + "\n</code>\n\nLabel (0 or 1):"
    if hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return sys_text + "\n\n" + user + " "

def enc(code: str):
    s = prompt_text(code)
    ids = tok(s, add_special_tokens=False).input_ids
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]
    attn = [1] * len(ids)
    return torch.tensor([ids], device=dev), torch.tensor([attn], device=dev)

def score(ids, attn, cand):
    c = torch.tensor([cand], device=dev)
    x = torch.cat([ids, c], dim=1)
    m = torch.cat([attn, torch.ones_like(c)], dim=1)
    logits = mdl(input_ids=x, attention_mask=m).logits
    n = ids.shape[1]
    s = 0.0
    for i, tok_id in enumerate(cand):
        pos = n + i - 1
        s += float(torch.log_softmax(logits[0, pos, :], dim=-1)[tok_id])
    return s

def predict(code: str):
    ids, attn = enc(code)
    lp0 = score(ids, attn, t0)
    lp1 = score(ids, attn, t1)
    m = max(lp0, lp1)
    p0 = math.exp(lp0 - m)
    p1 = math.exp(lp1 - m)
    p1 = p1 / (p0 + p1 + 1e-12)
    y = 1 if lp1 > lp0 else 0
    return y, p1

y = df["label"].to_numpy()
yp, ys = [], []
for code in tqdm(df["function"].tolist()):
    a, b = predict(code)
    yp.append(a)
    ys.append(b)

yp = np.array(yp, dtype=int)
ys = np.array(ys, dtype=float)

acc = accuracy_score(y, yp)
p, r, f1, _ = precision_recall_fscore_support(y, yp, average="binary", pos_label=1, zero_division=0)
cm = confusion_matrix(y, yp)
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
else:
    tn = fp = fn = tp = 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
mcc = matthews_corrcoef(y, yp)
kap = cohen_kappa_score(y, yp)
try:
    auc = roc_auc_score(y, ys)
except ValueError:
    auc = float("nan")

print("acc:", f"{acc:.4f}")
print("pre:", f"{p:.4f}")
print("rec:", f"{r:.4f}")
print("f1 :", f"{f1:.4f}")
print("spe:", f"{spec:.4f}")
print("mcc:", f"{mcc:.4f}")
print("kap:", f"{kap:.4f}")
print("auc:", f"{auc:.4f}" if not np.isnan(auc) else auc)

out = df.copy()
out["pred"] = yp
out["p1"] = ys
out_file = "qwen25_zeroshot_preds.csv"
out.to_csv(out_file, index=False)
print("saved:", out_file)

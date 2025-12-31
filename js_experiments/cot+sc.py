# -*- coding: utf-8 -*-
import os
import re
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

test_csv = "bb.csv"

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
local_dir = "qwen25_coder_1_5b_local"
save_local = True

seed = 42
max_ctx = 8192

n_samples = 5
temp = 0.7
top_p = 0.9
max_new = 6

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print("device:", dev)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(seed)

df = pd.read_csv(test_csv)
if "function" not in df.columns or "label" not in df.columns:
    raise ValueError("test_csv must have columns: function, label")

df["function"] = df["function"].astype(str)
df["label"] = df["label"].astype(int)

sys_text = """You are a senior security auditor for multi-language code (C/C++, Java, Kotlin, JavaScript/TypeScript, Python, Go, Rust, PHP, etc.).

Goal:
Classify whether the given single function is security-vulnerable.

Definitions:
1 = vulnerable
0 = not vulnerable

You MUST do a careful step-by-step security audit internally, but you MUST NOT reveal your reasoning.

Audit checklist (apply broadly across languages):
- Attacker control: can any input be influenced by an untrusted user, network, file, env var, IPC, request params, deserialization, headers, cookies, CLI args?
- Dangerous sinks: command execution, eval/dynamic code, SQL/NoSQL queries, template injection, path/file operations, SSRF, XXE, insecure deserialization, reflection, crypto misuse, auth/session, access control, unsafe redirects, XSS, injection, memory-unsafe patterns, race/TOCTOU.
- Missing safeguards: validation/sanitization, allowlists, encoding/escaping, parameterized queries, bounds checks, authentication/authorization checks, CSRF protection, proper cryptography (no hardcoded keys, weak RNG, no insecure modes), safe file path handling (no traversal), safe URL handling (no open redirect/SSRF), safe deserialization.
- Impact: could exploitation plausibly cause confidentiality/integrity/availability harm?

Decision rule:
Output 1 only if there is a plausible exploit path with meaningful security impact due to missing or insufficient safeguards.
Otherwise output 0.

Output constraints (STRICT):
Return exactly one character: 0 or 1.
No explanation. No extra text. No punctuation. No whitespace.
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

if getattr(mdl.config, "pad_token_id", None) is None:
    mdl.config.pad_token_id = tok.pad_token_id

if save_local and (not os.path.isdir(local_dir)):
    os.makedirs(local_dir, exist_ok=True)
    tok.save_pretrained(local_dir)
    mdl.save_pretrained(local_dir)

lab_re = re.compile(r"[01]")

def parse_lab(txt: str):
    t = (txt or "").strip()
    if t in ("0", "1"):
        return int(t)
    m = lab_re.search(t)
    return int(m.group(0)) if m else None

def prompt(code: str) -> str:
    user = (
        "Classify the following function.\n"
        "Think step-by-step internally, but output ONLY the final label.\n\n"
        "<code>\n"
        + code
        + "\n</code>\n\n"
        "Label (0 or 1):"
    )
    if getattr(tok, "chat_template", None) or hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return sys_text + "\n\n" + user

def enc(code: str):
    s = prompt(code)
    ids = tok(s, add_special_tokens=False).input_ids
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]
    attn = [1] * len(ids)
    return torch.tensor([ids], device=dev), torch.tensor([attn], device=dev)

@torch.inference_mode()
def draw_one(ids, attn, s: int):
    set_seed(s)
    out = mdl.generate(
        input_ids=ids,
        attention_mask=attn,
        do_sample=True,
        temperature=temp,
        top_p=top_p,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[0, ids.shape[1]:]
    txt = tok.decode(gen, skip_special_tokens=True)
    return parse_lab(txt), txt

@torch.inference_mode()
def greedy_one(ids, attn):
    out = mdl.generate(
        input_ids=ids,
        attention_mask=attn,
        do_sample=False,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[0, ids.shape[1]:]
    txt = tok.decode(gen, skip_special_tokens=True)
    lab = parse_lab(txt)
    return (lab if lab is not None else 0), txt

def predict(code: str, i: int):
    ids, attn = enc(code)
    votes, raws = [], []
    base = seed + 1337 * (i + 1)

    for k in range(n_samples):
        lab, txt = draw_one(ids, attn, base + k)
        raws.append(txt)
        if lab is not None:
            votes.append(lab)

    if not votes:
        lab, txt = greedy_one(ids, attn)
        votes = [lab]
        raws.append(txt)

    v1 = sum(votes)
    v0 = len(votes) - v1
    y = 1 if v1 > v0 else 0
    p1 = v1 / (len(votes) + 1e-12)
    return y, p1

y = df["label"].to_numpy()
yp, ys = [], []

print(f"running cot+sc: n_samples={n_samples} temp={temp} top_p={top_p}")
for i, code in enumerate(tqdm(df["function"].tolist())):
    a, b = predict(code, i)
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
out_file = "qwen25_cot_sc_preds.csv"
out.to_csv(out_file, index=False)
print("saved:", out_file)

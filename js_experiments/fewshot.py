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

prompt_csv = "datasets/npm_js_functions_prompt40.csv"
test_csv = "test_data.csv"

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
save_dir = "qwen2_5_coder_1_5b_instruct_local"
save_model_locally = True

n_few_shot = 4
seed = 42
max_ctx = 4096

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("device:", device)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

df_prompt = pd.read_csv(prompt_csv)
df_test = pd.read_csv(test_csv)

if "function" not in df_prompt.columns or "label" not in df_prompt.columns:
    raise ValueError("prompt_csv must have columns: function, label")
if "function" not in df_test.columns or "label" not in df_test.columns:
    raise ValueError("test_csv must have columns: function, label")

df_prompt["function"] = df_prompt["function"].astype(str)
df_test["function"] = df_test["function"].astype(str)
df_prompt["label"] = df_prompt["label"].astype(int)
df_test["label"] = df_test["label"].astype(int)

pos = df_prompt[df_prompt["label"] == 1]
neg = df_prompt[df_prompt["label"] == 0]

half = n_few_shot // 2
if len(pos) < half or len(neg) < half:
    raise ValueError(f"need at least {half} positive and {half} negative examples in prompt_csv")

few_shot_df = (
    pd.concat(
        [
            pos.sample(n=half, random_state=seed),
            neg.sample(n=half, random_state=seed + 1),
        ],
        ignore_index=True,
    )
    .sample(frac=1, random_state=seed)
    .reset_index(drop=True)
)

def build_few_shot_block(examples: pd.DataFrame) -> str:
    parts = []
    for i, r in examples.iterrows():
        parts.append(
            f"example {i+1}:\n"
            f"<code>\n{r['function']}\n</code>\n"
            f"label: {int(r['label'])}\n"
        )
    return "\n".join(parts)

few_shot_block = build_few_shot_block(few_shot_df)

instruction = """you are a security auditor.

task:
given a single function, output:
1 = vulnerable
0 = not vulnerable

general principle:
output 1 only if the function contains a plausible security weakness where an attacker could cause an unwanted security impact due to missing or insufficient safeguards.
output 0 otherwise.

how to decide (high-level):
- consider whether the function processes or depends on input that could be influenced by an attacker.
- consider whether the function performs any operation that could matter for security (e.g., decisions, access, execution, storage, communication, or exposure of data/resources).
- if attacker influence + insufficient safeguards could realistically lead to harm, output 1.
- if protections are adequate or there is not enough evidence of a real security risk, output 0.

output constraints (strict):
return exactly one character: 0 or 1.
no explanation. no extra text. no punctuation. no extra whitespace.
"""

prefix_text = f"""{instruction}

few-shot examples:
{few_shot_block}

now classify the following function:
<code>
"""

suffix_text = """
</code>

label (0 or 1): """

load_path = save_dir if os.path.isdir(save_dir) else model_name
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    load_path,
    torch_dtype=dtype,
    trust_remote_code=True,
)
model.to(device)
model.eval()
torch.set_grad_enabled(False)

if save_model_locally and (not os.path.isdir(save_dir)):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

cand0 = tokenizer("0", add_special_tokens=False).input_ids
cand1 = tokenizer("1", add_special_tokens=False).input_ids

def make_prompt_ids(code: str):
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids
    code_ids = tokenizer(code, add_special_tokens=False).input_ids

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    reserved = len(bos) + len(prefix_ids) + len(suffix_ids)
    max_code_len = max_ctx - reserved
    if max_code_len <= 0:
        raise RuntimeError("prompt too long for max_ctx; reduce few-shot or instruction")
    if len(code_ids) > max_code_len:
        code_ids = code_ids[:max_code_len]

    input_ids = bos + prefix_ids + code_ids + suffix_ids
    attn_mask = [1] * len(input_ids)

    return (
        torch.tensor([input_ids], device=device),
        torch.tensor([attn_mask], device=device),
    )

def score_candidate(prompt_ids, prompt_mask, candidate_ids):
    cand = torch.tensor([candidate_ids], device=device)
    input_ids = torch.cat([prompt_ids, cand], dim=1)
    attn_mask = torch.cat([prompt_mask, torch.ones_like(cand)], dim=1)

    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    prompt_len = prompt_ids.shape[1]

    total = 0.0
    for i, tok_id in enumerate(candidate_ids):
        pos = prompt_len + i - 1
        total += float(torch.log_softmax(logits[0, pos, :], dim=-1)[tok_id])
    return total

def classify_function(code: str):
    prompt_ids, prompt_mask = make_prompt_ids(code)
    lp0 = score_candidate(prompt_ids, prompt_mask, cand0)
    lp1 = score_candidate(prompt_ids, prompt_mask, cand1)

    m = max(lp0, lp1)
    p0 = math.exp(lp0 - m)
    p1 = math.exp(lp1 - m)
    p1 = p1 / (p0 + p1 + 1e-12)

    pred = 1 if lp1 > lp0 else 0
    return pred, p1

y_true = df_test["label"].to_numpy()
y_pred = []
y_score = []

for code in tqdm(df_test["function"].tolist()):
    pred, p1 = classify_function(code)
    y_pred.append(pred)
    y_score.append(p1)

y_pred = np.array(y_pred, dtype=int)
y_score = np.array(y_score, dtype=float)

def compute_metrics(y_true, y_pred, y_score):
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    out["precision"] = precision
    out["recall"] = recall
    out["f1"] = f1

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if np.unique(y_true).tolist() == [0]:
            tn = cm[0, 0]
        elif np.unique(y_true).tolist() == [1]:
            tp = cm[0, 0]

    out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    out["mcc"] = matthews_corrcoef(y_true, y_pred)
    out["kappa"] = cohen_kappa_score(y_true, y_pred)

    try:
        out["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        out["auc"] = float("nan")

    return out

metrics = compute_metrics(y_true, y_pred, y_score)

print("\nqwen2.5-coder-1.5b-instruct few-shot metrics")
for k, v in metrics.items():
    try:
        print(f"{k:12s}: {v:.4f}")
    except Exception:
        print(f"{k:12s}: {v}")

out_df = df_test.copy()
out_df["pred_label"] = y_pred
out_df["score_p1"] = y_score

out_path = "qwen2_5_coder_1_5b_fewshot_predictions.csv"
out_df.to_csv(out_path, index=False)
print("saved:", out_path)

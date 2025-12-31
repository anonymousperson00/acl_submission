# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


model_id = "meta-llama/Llama-3.2-3B-Instruct"
train_csv = "train.csv"
test_csv = "test.csv"
out_dir = "llama32_3b_qlora_bincls"

max_len = 512
bs = 4
epochs = 5
lr = 2e-4
seed = 42

set_seed(seed)


class CodeBinCls(Dataset):
    def __init__(self, df: pd.DataFrame, tok: AutoTokenizer, max_len: int):
        self.x = df["function"].astype(str).tolist()
        self.y = df["label"].astype(int).tolist()
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i: int):
        enc = self.tok(
            self.x[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    p = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pos = p[:, 1]

    acc = accuracy_score(labels, preds)
    pre, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    mcc = matthews_corrcoef(labels, preds)
    kap = cohen_kappa_score(labels, preds)

    try:
        auc = roc_auc_score(labels, pos)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "specificity": spe,
        "mcc": mcc,
        "kappa": kap,
        "auc": auc,
    }


def main():
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)

    if "function" not in tr.columns or "label" not in tr.columns:
        raise ValueError("train.csv must have columns: function, label")
    if "function" not in te.columns or "label" not in te.columns:
        raise ValueError("test.csv must have columns: function, label")

    tr["label"] = tr["label"].astype(int)
    te["label"] = te["label"].astype(int)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    mdl.config.pad_token_id = tok.pad_token_id
    mdl.config.use_cache = False

    mdl = prepare_model_for_kbit_training(mdl)

    lora = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    mdl = get_peft_model(mdl, lora)
    mdl.print_trainable_parameters()

    ds_tr = CodeBinCls(tr, tok, max_len)
    ds_te = CodeBinCls(te, tok, max_len)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trn = Trainer(
        model=mdl,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_te,
        tokenizer=tok,
        compute_metrics=metrics_fn,
    )

    trn.train()
    m = trn.evaluate(ds_te)

    print("llama32_3b_qlora_test")
    for k, v in m.items():
        print(f"{k}: {v}")

    trn.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()

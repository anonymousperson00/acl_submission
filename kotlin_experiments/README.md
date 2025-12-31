### Base model
- Models: Code PTMs
- Task: **binary sequence classification** (`num_labels=2`)
- Input field: `function`
- Label field: `label` (0/1)

### Quantization (BitsAndBytes)
- 4-bit loading: `load_in_4bit=True`
- Quant type: `bnb_4bit_quant_type="nf4"`
- Double quant: `bnb_4bit_use_double_quant=True`
- Compute dtype: `bnb_4bit_compute_dtype=torch.bfloat16`
- Model dtype: `torch_dtype=torch.bfloat16`
- Device placement: `device_map="auto"`

### LoRA adapter setup
- `r=64`
- `lora_alpha=16`
- `lora_dropout=0.05`
- `bias="none"`
- `task_type="SEQ_CLS"`
- Target modules:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  - `gate_proj`, `up_proj`, `down_proj`

### Tokenization / sequence length
- `max_len=512`
- Padding: `padding="max_length"`
- Truncation: `truncation=True`
- `pad_token` set to `eos_token` if missing 

### Training hyperparameters (TrainingArguments)
- Output: `out_dir="xxxx"`
- Batch size:
  - train: `per_device_train_batch_size=4`
  - eval:  `per_device_eval_batch_size=4`
- Epochs: `num_train_epochs=5`
- Learning rate: `learning_rate=2e-4`
- Weight decay: `weight_decay=0.01`
- LR scheduler: `lr_scheduler_type="cosine"`
- Warmup: `warmup_ratio=0.03`
- Gradient checkpointing: `gradient_checkpointing=True`
- Precision: `bf16=True`
- Evaluation: `eval_strategy="epoch"`
- Saving: `save_strategy="epoch"`, `save_total_limit=2`
- Logging: `logging_steps=50`
- Seed: `seed=42`
- Reporting: `report_to="none"`

### Evaluation metrics
- Accuracy, Precision, Recall, F1
- Specificity (from confusion matrix)
- MCC, Cohen’s Kappa
- AUC
- Detailed Results are included in the Paper both for finetuning and cross language analysis.

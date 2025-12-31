# acl_submission


## Kotlin Based
- **Kotlin snippet/class corpus builder**: collects Kotlin security-related samples and assigns a stable `sample_id` (e.g., `KOT_000123`) so every file/snippet can be traced across labeling + modeling steps.
- **Static labeling (Semgrep Kotlin OWASP)**:
  - runs Semgrep with a Kotlin OWASP ruleset on the snippet directory,
  - aggregates findings per `sample_id` (rule ids, severity counts, CWE/OWASP tags),
  - merges aggregated results back into the main Kotlin CSV to produce labeled datasets (rule-hit counts + binary vuln flags).
- **Train/Test preparation**: builds train/test splits from the labeled Kotlin dataset (optionally de-duplicated by `sample_id` or normalized code hash) for reproducible experiments.
- **LLM fine-tuning (supervised)**:
  - fine-tunes code models on Kotlin labeled data (binary classification: vulnerable vs non-vulnerable),
  - supports LoRA/QLoRA settings for GPU-efficient training,
  - results and cross language analysis recorded in the tables of the paper.


## GHSA Based
- **GHSA → package-level metadata export** (up to 15,000 advisories; expanded to package rows).
- **Patch-based dataset builder**: clones public GitHub repos, locates fixing commits, extracts *before/after* code (full file + localized patch snippets), and keeps only samples that parse successfully via a JS/TS AST check.
- **LLM evaluation scripts** (e.g., Qwen2.5-Coder-1.5B-Instruct) for:
  - **Zero-shot**
  - **Few-shot**
  - **CoT + Self-Consistency**
  - results and cross language analysis recorded in the tables of the paper
    

## Ebvironmental Setup
```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
